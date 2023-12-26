import functools
from typing import Optional, Dict, List
from graphcast import autoregressive
from graphcast import casting
from graphcast import graphcast
from graphcast import normalization
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import numpy as np
import xarray as xa
import hydra, dataclasses

# Build jitted functions, and possibly initialize random weights

def drop_state(fn):
	return lambda **kw: fn(**kw)[0]
def construct_wrapped_graphcast( model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig, norms: Dict[str,xa.Dataset]):
	"""Constructs and wraps the GraphCast Predictor."""
	# Deeper one-step predictor.
	predictor = graphcast.GraphCast(model_config, task_config)

	# Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
	# from/to float32 to/from BFloat16.
	predictor = casting.Bfloat16Cast(predictor)
#	print( f"\n **** Norm (std) Data vars = {list(stddev_by_level.data_vars.keys())}")

	# Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
	# BFloat16 happens after applying normalization to the inputs/targets.
	predictor = normalization.InputsAndResiduals(
	  predictor,
	  diffs_stddev_by_level=norms['diffs_stddev_by_level'],
	  mean_by_level=norms['mean_by_level'],
	  stddev_by_level=norms['stddev_by_level'])

	# Wraps everything so the one-step model can produce trajectories.
	predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
	return predictor

@hk.transform_with_state
def run_forward(model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig, norms: Dict[str,xa.Dataset], inputs: xa.Dataset, targets_template: xa.Dataset, forcings: xa.Dataset):
	predictor = construct_wrapped_graphcast(model_config, task_config, norms )
	return predictor(inputs, targets_template=targets_template, forcings=forcings)

@hk.transform_with_state
def loss_fn(model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig, norms: Dict[str,xa.Dataset], inputs: xa.Dataset, targets: xa.Dataset, forcings: xa.Dataset):
	from graphcast.losses import is_tracer
	print(f"\n --------------------------- loss_fn: --------------------------- ")
	print( f" *** inputs ({type(inputs)}):")
	for vn, dv in inputs.data_vars.items():
		print(f" > {vn}{dv.dims}: {dv.shape}, is_tracer = {is_tracer(dv)}")
	print( f" *** targets ({type(targets)}):")
	for vn, dv in targets.data_vars.items():
		print(f" > {vn}{dv.dims}: {dv.shape}, is_tracer = {is_tracer(dv)}")
	print( f" *** forcings ({type(forcings)}):")
	for vn, dv in forcings.data_vars.items():
		print(f" > {vn}{dv.dims}: {dv.shape}, is_tracer = {is_tracer(dv)}")
	predictor = construct_wrapped_graphcast(model_config, task_config, norms)
	loss, diagnostics = predictor.loss(inputs, targets, forcings)
	return xarray_tree.map_structure(
	  lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True), (loss, diagnostics))

def grads_fn(params: Dict, state: Dict, model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig, norms: xa.Dataset, inputs: xa.Dataset, targets: xa.Dataset, forcings: xa.Dataset):
	def _aux(params_, state_, i, t, f):
		(loss_, diagnostics_), next_state_ = loss_fn.apply(  params_, state_, jax.random.PRNGKey(0), model_config, task_config, norms, i, t, f)
		return loss_, (diagnostics_, next_state_)
	(loss, (diagnostics, next_state)), grads = jax.value_and_grad( _aux, has_aux=True)(params, state, inputs, targets, forcings)
	return loss, diagnostics, next_state, grads
