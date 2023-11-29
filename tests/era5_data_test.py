import functools
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from fmgraphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
from fmbase.util.ops import format_timedeltas, print_dict
import haiku as hk
import jax, time
from fmbase.util.ops import fmbdir
import numpy as np
import xarray
import hydra, dataclasses
from fmbase.util.config import configure, cfg
from typing import List, Union, Tuple, Optional, Dict, Type

hydra.initialize( version_base=None, config_path="../config" )
configure( 'explore-era5' )

# Load the model

root = fmbdir('model')
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

print( "\n -- TaskConfig --")
print( f" * input_variables   = {task_config.input_variables}")
print( f" * target_variables  = {task_config.target_variables}")
print( f" * forcing_variables = {task_config.forcing_variables}")
print( f" * pressure_levels   = {task_config.pressure_levels}")


print("\nAll Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)

for (title,dset) in [ ('train',train_inputs), ('target',train_targets), ('forcing',train_forcings) ]:
	nfeatures = 0
	print(f"\n{title} inputs:   ")
	for vname, dvar in dset.data_vars.items():
		ndvar: np.ndarray = dvar.values
		nfeatures = nfeatures + (1 if ndvar.ndim == 4 else ndvar.shape[2])
		print(f" > {vname}{dvar.dims}: shape: {dvar.shape}, dtype: {dvar.dtype}, range: ({ndvar.min():.3f},{ndvar.max():.3f}), mean,std: ({ndvar.mean():.3f},{ndvar.std():.3f})")
	print( f" ---------- N Features: {nfeatures}  ---------- ")

with open(f"{root}/stats/diffs_stddev_by_level.nc","rb") as f:
	diffs_stddev_by_level: xarray.Dataset = xarray.load_dataset(f).compute()
with open(f"{root}/stats/mean_by_level.nc","rb") as f:
	mean_by_level: xarray.Dataset = xarray.load_dataset(f).compute()
with open(f"{root}/stats/stddev_by_level.nc","rb") as f:
	stddev_by_level: xarray.Dataset = xarray.load_dataset(f).compute()

print( "\n * Normalization data: mean_by_level * ")
for k,v in mean_by_level.data_vars.items():
	print( f" ** {k}[{v.size}]")

coords = train_inputs.data_vars['temperature'].coords

print( f"\n Coords: ")
print( f"\n ---> Lat:   {coords['lat'].values.tolist()}")
print( f"\n ---> Lon:   {coords['lon'].values.tolist()}")
print( f"\n ---> Level: {coords['level'].values.tolist()}")

def with_configs(fn):
	return functools.partial( fn, model_config=model_config, task_config=task_config )

def with_params(fn):
	return functools.partial(fn, params=params, state=state)

def construct_wrapped_graphcast( modelconfig: graphcast.ModelConfig, taskconfig: graphcast.TaskConfig):
	predictor = graphcast.GraphCast(modelconfig, taskconfig)
	predictor = casting.Bfloat16Cast(predictor)
	predictor = normalization.InputsAndResiduals( predictor, diffs_stddev_by_level=diffs_stddev_by_level, mean_by_level=mean_by_level, stddev_by_level=stddev_by_level)
	predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
	return predictor

@hk.transform_with_state
def run_forward(modelconfig, taskconfig, inputs, targets_template, forcings):
	predictor = construct_wrapped_graphcast(modelconfig, taskconfig)
	print( f"\n Run forward-> inputs:")
	for vn, dv in inputs.data_vars.items():
		print(f" > {vn}{dv.dims}: {dv.shape}")
	print( f"\n Run forward-> targets_template:")
	for vn, dv in targets_template.data_vars.items():
		print(f" > {vn}{dv.dims}: {dv.shape}")
	return predictor(inputs, targets_template=targets_template, forcings=forcings)

init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
	params, state = init_jitted( rng=jax.random.PRNGKey(0), inputs=train_inputs, targets_template=train_targets, forcings=train_forcings)

print( f"Weights:" )
for k,v in params.items():
	if 'w' in v.keys():
		print( f" >> {k}: {v['w'].shape}")

