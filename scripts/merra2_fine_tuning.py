from fmbase.source.merra2.model import load_batch
from fmgraphcast.data_utils import load_merra2_norm_data
from fmgraphcast.config import save_params, load_params
import xarray as xa
import functools
from graphcast import autoregressive
from graphcast import casting
from fmgraphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax, time
import numpy as np
import hydra, dataclasses
from datetime import date
from fmbase.util.dates import date_list
from fmbase.util.config import configure, cfg
from typing import List, Union, Tuple, Optional, Dict, Type

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-finetuning' )
t0 = time.time()

def parse_file_parts(file_name):
	return dict(part.split("-", 1) for part in file_name.split("_"))

def dtypes( d: Dict ):
	return { k: type(v) for k,v in d.items() }

res,levels,steps = cfg().model.res,  cfg().model.levels,  cfg().model.steps
year, month, day =  cfg().model.year,  cfg().model.month,  cfg().model.day
train_steps, eval_steps = cfg().task.train_steps, cfg().task.eval_steps
runid = "small"
(params, model_config, task_config) = load_params("merra2", runid=runid, hydra_config=False )
state = {}
lr = cfg().task.lr

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load MERRA2 Data
#-----------------

dts         = cfg().task.data_timestep
start = date( cfg().task.year, cfg().task.month, cfg().task.day )
ndays = 4
target_lead_times = [ f"{iS*dts}h" for iS in range(1,train_steps+1) ]
eval_lead_times =   [ f"{iS*dts}h" for iS in range(1,eval_steps+1) ]

print( "  --------------------- MERRA2 ---------------------")
example_batch: xa.Dataset = load_batch( date_list(start,ndays), cfg().task )

print("Loaded Batch:")
for vname, dvar in example_batch.data_vars.items():
	print( f" {vname}{list(dvar.dims)}: shape={dvar.shape}")

norm_data: Dict[str,xa.Dataset] = load_merra2_norm_data()

itf = data_utils.extract_inputs_targets_forcings( example_batch, target_lead_times=target_lead_times, **dataclasses.asdict(task_config) )
train_inputs, train_targets, train_forcings = itf

itf = data_utils.extract_inputs_targets_forcings( example_batch, target_lead_times=eval_lead_times, **dataclasses.asdict(task_config) )
eval_inputs, eval_targets, eval_forcings = itf

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)

print("Eval Inputs:   ", eval_inputs.dims.mapping)
for vname, dvar in eval_inputs.data_vars.items():
	print( f" > {vname}{dvar.dims}: {dvar.shape}")
	if "time" in dvar.dims:
		print(f" --> time: {dvar.coords['time'].values.tolist()}")

print("Eval Targets:  ", eval_targets.dims.mapping)
for vname, dvar in eval_targets.data_vars.items():
	print( f" > {vname}{dvar.dims}: {dvar.shape}")
	if "time" in dvar.dims:
		print(f" --> time: {dvar.coords['time'].values.tolist()}")
print("Eval Forcings: ", eval_forcings.dims.mapping)

print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

def construct_wrapped_graphcast( model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig):
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
	  diffs_stddev_by_level=norm_data['diffs_stddev_by_level'],
	  mean_by_level=norm_data['mean_by_level'],
	  stddev_by_level=norm_data['stddev_by_level'])

	# Wraps everything so the one-step model can produce trajectories.
	predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
	return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
	predictor = construct_wrapped_graphcast(model_config, task_config)
	print( f"\n Run forward-> inputs:")
	for vn, dv in inputs.data_vars.items():
		print(f" > {vn}{dv.dims}: {dv.shape}")
	print( f"\n Run forward-> targets_template:")
	for vn, dv in targets_template.data_vars.items():
		print(f" > {vn}{dv.dims}: {dv.shape}")
	return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
	predictor = construct_wrapped_graphcast(model_config, task_config)
	loss, diagnostics = predictor.loss(inputs, targets, forcings)
	return xarray_tree.map_structure(
	  lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True), (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
	def _aux(params, state, i, t, f):
		(loss, diagnostics), next_state = loss_fn.apply(  params, state, jax.random.PRNGKey(0), model_config, task_config, i, t, f)
		return loss, (diagnostics, next_state)
	(loss, (diagnostics, next_state)), grads = jax.value_and_grad( _aux, has_aux=True)(params, state, inputs, targets, forcings)
	return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
	return functools.partial( fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
	return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
	return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
	params, state = init_jitted( rng=jax.random.PRNGKey(0), inputs=train_inputs, targets_template=train_targets, forcings=train_forcings)

loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = jax.jit(with_configs(grads_fn))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
  "Model resolution doesn't match the data resolution. You likely want to re-filter the dataset list, and download the correct data.")

print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

nepochs = cfg().task.nepochs
for epoch in range(nepochs):
	te= time.time()
	loss, diagnostics, next_state, grads = with_params(grads_fn_jitted)( inputs=train_inputs, targets=train_targets, forcings=train_forcings )
	mean_grad = np.mean( jax.tree_util.tree_flatten( jax.tree_util.tree_map( lambda x: np.abs(x).mean(), grads ) )[0] )
	max_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).max(), grads))[0])
	params = jax.tree_map(  lambda p, g: p - lr * g, params, grads)
	print(f" * EPOCH {epoch}: Loss= {loss:.6f}, Mean/Max |dW|= {lr*mean_grad:.6f} / {lr*max_grad:.6f}, comptime= {time.time()-te:.1f} sec")

save_params( params, model_config, task_config, runid=runid )

# predictions: xarray.Dataset = rollout.chunked_prediction( run_forward_jitted, rng=jax.random.PRNGKey(0), inputs=eval_inputs,
# 														        targets_template=eval_targets * np.nan, forcings=eval_forcings)
#
# print( f" ***** Completed forecast, result variables:  ")
# for vname, dvar in predictions.data_vars.items():
# 	print( f" > {vname}{dvar.dims}: {dvar.shape}")
# 	ndvar: np.ndarray = dvar.values
# 	tvar: Optional[xarray.DataArray] = dvar.coords.get('time')
# 	print(f"   --> dtype: {dvar.dtype}, range: ({ndvar.min():.3f},{ndvar.max():.3f}), mean,std: ({ndvar.mean():.3f},{ndvar.std():.3f}), time: {format_timedeltas(tvar)}")
#
#
# print( f"Completed in {time.time()-t0} sec.")
#
#
