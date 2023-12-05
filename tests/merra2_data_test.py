from fmbase.source.merra2.model import YearMonth, load_batch
from fmgraphcast.data_utils import load_merra2_norm_data
from fmgraphcast.config import hydra_config_files
import xarray as xa
import functools
from graphcast import autoregressive
from graphcast import casting
from fmgraphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
import haiku as hk
import jax, time
import numpy as np, pandas as pd
import xarray
import hydra, dataclasses
from fmbase.util.config import configure, cfg
from typing import List, Union, Tuple, Optional, Dict, Type

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-finetuning' )
t0 = time.time()

def parse_file_parts(file_name):
	return dict(part.split("-", 1) for part in file_name.split("_"))

def dtypes( d: Dict ):
	return { k: type(v) for k,v in d.items() }

params, state = None, None
res,levels,steps = cfg().model.res,  cfg().model.levels,  cfg().model.steps
year, month, day =  cfg().model.year,  cfg().model.month,  cfg().model.day


train_steps, eval_steps = cfg().task.train_steps, cfg().task.eval_steps
(model_config,task_config) = hydra_config_files()
ndays = 3
lr = cfg().task.lr

print( "\n -- TaskConfig --")
print( f" * input_variables   = {task_config.input_variables}")
print( f" * target_variables  = {task_config.target_variables}")
print( f" * forcing_variables = {task_config.forcing_variables}")
print( f" * pressure_levels   = {task_config.pressure_levels}")

print( "\n -- ModelConfig --")
print( f" * mesh_size      = {model_config.mesh_size}")
print( f" * gnn_msg_steps  = {model_config.gnn_msg_steps}")
print( f" * latent_size    = {model_config.latent_size}")
print( f" * hidden_layers  = {model_config.hidden_layers}")
print( f" * resolution     = {model_config.resolution}")
print( f" * radius_qfel    = {model_config.radius_query_fraction_edge_length}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load MERRA2 Data
#-----------------

dts         = cfg().task.data_timestep
target_lead_times = slice("6h", f"{train_steps*6}h") # [ f"{iS*dts}h" for iS in range(1,train_steps+1) ]
eval_lead_times =   slice("6h", f"{eval_steps*6}h") #[ f"{iS*dts}h" for iS in range(1,eval_steps+1) ]

print( "  --------------------- MERRA2 ---------------------")
example_batch: xa.Dataset = load_batch( year, month, day, ndays, cfg().task )
vtime: List[str] = [str(pd.Timestamp(dt64)) for dt64 in example_batch.coords['time'].values.tolist()]
print(f"\n -------> batch time: {vtime}\n")

print("\nLoaded Batch:")
for vname in example_batch.data_vars.keys():
	dvar = example_batch.data_vars[vname]
	print( f" {vname}{list(dvar.dims)}: shape={dvar.shape}")

norm_data: Dict[str,xa.Dataset] = load_merra2_norm_data()

t0 = time.time()
itf = data_utils.extract_inputs_targets_forcings( example_batch, target_lead_times=target_lead_times, **dataclasses.asdict(task_config) )
train_inputs, train_targets, train_forcings = itf

itf = data_utils.extract_inputs_targets_forcings( example_batch, target_lead_times=eval_lead_times, **dataclasses.asdict(task_config) )
eval_inputs, eval_targets, eval_forcings = itf

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
	for vname in dset.data_vars.keys():
		dvar = dset.data_vars[vname]
		ndvar: np.ndarray = dvar.values
		nfeatures = nfeatures + (ndvar.shape[2] if (ndvar.ndim == 5) else 1)
		print(f" > {vname}{dvar.dims}: shape: {dvar.shape}, dtype: {dvar.dtype}, range: ({ndvar.min():.3f},{ndvar.max():.3f}), mean,std: ({ndvar.mean():.3f},{ndvar.std():.3f})")
	print( f" ---------- N Features: {nfeatures}  ---------- ")

norm_data: Dict[str,xa.Dataset] = load_merra2_norm_data()
diffs_stddev_by_level: xarray.Dataset = norm_data['diffs_stddev_by_level']
mean_by_level: xarray.Dataset =  norm_data['mean_by_level']
stddev_by_level: xarray.Dataset =  norm_data['stddev_by_level']

coords = train_inputs.data_vars['temperature'].coords

print( f"\n Coords: ")
print( f"\n ---> Lat:   {coords['lat'].values.tolist()}")
print( f"\n ---> Lon:   {coords['lon'].values.tolist()}")
print( f"\n ---> Level: {coords['level'].values.tolist()}")

def with_configs(fn):
	return functools.partial( fn, modelconfig=model_config, taskconfig=task_config )

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
	for vn in inputs.data_vars.keys():
		dv = inputs.data_vars[vn]
		print(f" > {vn}{dv.dims}: {dv.shape}")
	return predictor(inputs, targets_template=targets_template, forcings=forcings)

t1 = time.time()

init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
	params, state = init_jitted( rng=jax.random.PRNGKey(0), inputs=train_inputs, targets_template=train_targets, forcings=train_forcings)

print( f"Computed Weights in {(time.time()-t1):.2f} {(t1-t0):.2f} sec:" )
for k,v in params.items():
	if 'w' in v.keys():
		print( f" >> {k}: {v['w'].shape}")

