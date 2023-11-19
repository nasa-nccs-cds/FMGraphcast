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
from fmbase.util.config import configure, cfg

# Build jitted functions, and possibly initialize random weights

def construct_wrapped_graphcast( model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig, norm_data: Dict[str,xa.Dataset]):
	"""Constructs and wraps the GraphCast Predictor."""
	# Deeper one-step predictor.
	predictor = graphcast.GraphCast(model_config, task_config)

	# Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
	# from/to float32 to/from BFloat16.
	predictor = casting.Bfloat16Cast(predictor)

	# Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
	# BFloat16 happens after applying normalization to the inputs/targets.
	print( f"\n **** Norm (std) Data vars = {list(norm_data['stddev_by_level'].data_vars.keys())}")
	predictor = normalization.InputsAndResiduals( predictor, **norm_data )

	# Wraps everything so the one-step model can produce trajectories.
	predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
	return predictor

@hk.transform_with_state
def run_forward(model_config, task_config, inputs: xa.Dataset, targets_template: xa.Dataset, forcings: xa.Dataset, norm_data: Dict[str,xa.Dataset] ):
	predictor = construct_wrapped_graphcast(model_config, task_config, norm_data)
	print( f"\n Run forward-> inputs:")
	for vn, dv in inputs.data_vars.items():
		print(f" > {vn}{dv.dims}: {dv.shape}")
	print( f"\n Run forward-> targets_template:")
	for vn, dv in targets_template.data_vars.items():
		print(f" > {vn}{dv.dims}: {dv.shape}")
	print( f"\n Run forward-> targets_template:")
	for vn, dv in targets_template.data_vars.items():
		print(f" > {vn}{dv.dims}: {dv.shape}")
	return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings, norm_data: Dict[str,xa.Dataset]):
	predictor = construct_wrapped_graphcast(model_config, task_config, norm_data)
	loss, diagnostics = predictor.loss(inputs, targets, forcings)
	return xarray_tree.map_structure(
	  lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True), (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
	def _aux(params, state, i, t, f):
		(loss, diagnostics), next_state = loss_fn.apply(  params, state, jax.random.PRNGKey(0), model_config, task_config, i, t, f)
		return loss, (diagnostics, next_state)
	(loss, (diagnostics, next_state)), grads = jax.value_and_grad( _aux, has_aux=True)(params, state, inputs, targets, forcings)
	return loss, diagnostics, next_state, grads


# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
	return lambda **kw: fn(**kw)[0]
