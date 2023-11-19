import functools
from graphcast import checkpoint
from fmgraphcast import data_utils
from graphcast import graphcast
from fmgraphcast.model import run_forward, grads_fn, loss_fn
from graphcast import rollout
from fmbase.util.ops import format_timedeltas, print_dict
import haiku as hk
import jax, time
import numpy as np
import xarray
import hydra, dataclasses
from fmbase.util.config import configure, cfg
from typing import List, Union, Tuple, Optional, Dict, Type

hydra.initialize( version_base=None, config_path="../config" )
configure( 'explore-era5' )

# Load the model

root = cfg().platform.model.format( **cfg().platform ).format( **cfg().platform )
params_file = cfg().task.params
pfilepath = f"{root}/params/{params_file}.npz"
print( f" root = ", root )
print( f" params_file = ", params_file )
print( f" pfilepath = ", pfilepath )
year = 2022
t0 = time.time()

with open(pfilepath, "rb") as f:
	ckpt = checkpoint.load(f, graphcast.CheckPoint)
	params = ckpt.params
	state = {}

	model_config = ckpt.model_config
	task_config = ckpt.task_config
	print("Model description:\n", ckpt.description, "\n")
	print_dict( "model_config", model_config )
	print_dict("task_config", task_config )

# Load weather data

res,levels,steps = cfg().model.res,  cfg().model.levels,  cfg().model.steps
month, day =  cfg().model.month,  cfg().model.day
dataset_file = f"{root}/data/era5/res-{res}_levels-{levels}_steps-{steps:0>2}/{year}-{month:0>2}-{day:0>2}.nc"

with open(dataset_file,"rb") as f:
	example_batch = xarray.load_dataset(f).compute()

# Extract training and eval data

train_steps, eval_steps = cfg().task.train_steps, cfg().task.eval_steps
train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings( example_batch,
												target_lead_times=slice("6h", f"{train_steps*6}h"), **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings( example_batch,
												target_lead_times=slice("6h", f"{eval_steps*6}h"), **dataclasses.asdict(task_config))

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)

print("\nEval Inputs:   ", eval_inputs.dims.mapping)
print_dict( "DSET attrs", eval_inputs.attrs )
for vname, dvar in eval_inputs.data_vars.items():
	ndvar: np.ndarray = dvar.values
	print(f" > {vname}{dvar.dims}: {dvar.shape}")
	tvar: Optional[xarray.DataArray] = dvar.coords.get('time')
	print(f"   --> dtype: {dvar.dtype}, range: ({ndvar.min():.3f},{ndvar.max():.3f}), mean,std: ({ndvar.mean():.3f},{ndvar.std():.3f}), time: {format_timedeltas(tvar)}")

print("\nEval Targets:  ", eval_targets.dims.mapping)
for vname, dvar in eval_targets.data_vars.items():
	print( f" > {vname}{dvar.dims}: {dvar.shape}")
	tvar: Optional[xarray.DataArray] = dvar.coords.get('time')
	print(f"   --> time: {format_timedeltas(tvar)}")
print("\nEval Forcings: ", eval_forcings.dims.mapping)

# Load normalization data

norm_data = {}
with open(f"{root}/stats/diffs_stddev_by_level.nc","rb") as f:
	norm_data['diffs_stddev_by_level'] = xarray.load_dataset(f).compute()
with open(f"{root}/stats/mean_by_level.nc","rb") as f:
	norm_data['mean_by_level'] = xarray.load_dataset(f).compute()
with open(f"{root}/stats/stddev_by_level.nc","rb") as f:
	norm_data['stddev_by_level'] = xarray.load_dataset(f).compute()

print( " * Loaded normalization data * ")

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
	return functools.partial( fn, model_config=model_config, task_config=task_config, norm_data=norm_data )

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
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

# Autoregressive rollout (loop in python)

assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
  "Model resolution doesn't match the data resolution. You likely want to re-filter the dataset list, and download the correct data.")

print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

predictions: xarray.Dataset = rollout.chunked_prediction( run_forward_jitted, rng=jax.random.PRNGKey(0), inputs=eval_inputs,
														        targets_template=eval_targets * np.nan, forcings=eval_forcings)
#
# print( f" ***** Completed forecast, result variables:  ")
# for vname, dvar in predictions.data_vars.items():
# 	print( f" > {vname}{dvar.dims}: {dvar.shape}")
# 	ndvar: np.ndarray = dvar.values
# 	tvar: Optional[xarray.DataArray] = dvar.coords.get('time')
# 	print(f"   --> dtype: {dvar.dtype}, range: ({ndvar.min():.3f},{ndvar.max():.3f}), mean,std: ({ndvar.mean():.3f},{ndvar.std():.3f}), time: {format_timedeltas(tvar)}")
#
# print( f"Completed in {time.time()-t0} sec.")
#
#
