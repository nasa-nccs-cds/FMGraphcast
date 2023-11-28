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
from fmgraphcast.config import model_config, task_config, norm_data, cparms

# Build jitted functions, and possibly initialize random weights

def construct_wrapped_graphcast( **kwargs ):
	"""Constructs and wraps the GraphCast Predictor."""
	# Deeper one-step predictor.
	predictor = graphcast.GraphCast( kwargs['model_config'], kwargs['task_config'])

	# Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
	# from/to float32 to/from BFloat16.
	predictor = casting.Bfloat16Cast(predictor)

	# Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
	# BFloat16 happens after applying normalization to the inputs/targets.
	# print( f"\n **** Norm (std) Data vars = {list(norm_data['stddev_by_level'].data_vars.keys())}")
	predictor = normalization.InputsAndResiduals( predictor, **kwargs['norm_data'] )

	# Wraps everything so the one-step model can produce trajectories.
	predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
	return predictor

@hk.transform_with_state
def run_forward(inputs: xa.Dataset, targets_template: xa.Dataset, forcings: xa.Dataset, **kwargs ):
	predictor = construct_wrapped_graphcast(**kwargs)
	print( f"\n Run forward-> inputs:")
	for vn, dv in inputs.data_vars.items():
		print(f" > {vn}{dv.dims}: {dv.shape}")
	print( f"\n Run forward-> targets_template:")
	for vn, dv in targets_template.data_vars.items():
		print(f" > {vn}{dv.dims}: {dv.shape}")
	return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(inputs: xa.Dataset, targets: xa.Dataset, forcings: xa.Dataset, **kwargs):
	predictor = construct_wrapped_graphcast(**kwargs)
	loss, diagnostics = predictor.loss(inputs, targets, forcings)
	return xarray_tree.map_structure(
	  lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True), (loss, diagnostics))

@jax.jit
def grads_fn(inputs: xa.Dataset, targets: xa.Dataset, forcings: xa.Dataset, **kwargs ):
	def _aux(i, t, f, **kw):
		(loss1, diagnostics1), next_state1 = loss_fn.apply( kw['params'], kw['state'], jax.random.PRNGKey(0), i, t, f, **kw)
		return loss1, (diagnostics1, next_state1)
	(loss, (diagnostics, next_state)), grads = jax.value_and_grad( _aux, has_aux=True)(inputs, targets, forcings, **kwargs)
	return loss, diagnostics, next_state, grads

def with_configs(fn):
	return functools.partial( fn, model_config=model_config, task_config=task_config, norm_data=norm_data )

# Always pass params and state, so the usage below are simpler
def with_params(fn):
	return functools.partial(fn, params=params, state=state)
@jax.jit
def update_fn( inputs: xa.Dataset, targets: xa.Dataset, forcings: xa.Dataset, **kwargs ):
	lr = cfg().task.lr
	grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
	loss1, diagnostics1, next_state, grads = grads_fn_jitted(inputs=inputs,targets=targets,forcings=forcings)
	mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
	mean_loss = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), loss1))[0])
	next_params = jax.tree_map(  lambda p, g: p - lr * g, kwargs['params'], grads)
	print( f" Update: mean_grad = {mean_grad}, mean_loss = {mean_loss}")
	return next_params, next_state

def train_model( inputs: xa.Dataset, targets: xa.Dataset, forcings: xa.Dataset ):
	nepochs = cfg().task.nepochs
	params, state = run_forward.init(rng=jax.random.PRNGKey(0), inputs=inputs, targets_template=targets, forcings=forcings, **cparms())
	for epoch in range(nepochs):
		params, state = update_fn( inputs, targets, forcings, params=params, state=state, **cparms())


# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
	return lambda **kw: fn(**kw)[0]
