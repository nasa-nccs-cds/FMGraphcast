import functools
from typing import Optional, Dict
from fmbase.source.merra2.model import YearMonth, load_batch
from fmbase.source.merra2.preprocess import load_norm_data
from fmbase.util.config import config_files
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import numpy as np
import xarray as xa
import hydra, dataclasses
from fmbase.util.config import configure, cfg

hydra.initialize( version_base=None, config_path="../config" )
configure( 'explore-era5' )

def parse_file_parts(file_name):
	return dict(part.split("-", 1) for part in file_name.split("_"))

res,levels,steps = cfg().model.res,  cfg().model.levels,  cfg().model.steps
year, month, day =  cfg().model.year,  cfg().model.month,  cfg().model.day
train_steps, eval_steps = cfg().task.train_steps, cfg().task.eval_steps
(model_config,task_config) = config_files()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load MERRA2 Data
#-----------------

params = None
state = {}
dts         = cfg().task.data_timestep
coords = dict( z="level", x="lon", y="lat" )
start = YearMonth(year,month)
end = YearMonth(year,month+1)
target_lead_times = [ f"{iS*dts}h" for iS in range(1,train_steps+1) ]
eval_lead_times =   [ f"{iS*dts}h" for iS in range(1,eval_steps+1) ]

print( "  --------------------- MERRA2 ---------------------")
example_batch: xa.Dataset = load_batch( start, end, cfg().task, coords=coords )

print("Loaded Batch:")
for vname, dvar in example_batch.data_vars.items():
	print( f" {vname}{list(dvar.dims)}: shape={dvar.shape}")

norm_data: Dict[str,xa.Dataset] = load_norm_data( cfg().task.dataset_version )

print("Loaded Norm Data:")
for vname, ndset in norm_data.items():
	for nname, ndata in ndset.data_vars.items():
		print( f" {vname}.{nname}: shape={ndata.shape}")

train_inputs, train_targets, train_forcings = \
	data_utils.extract_inputs_targets_forcings( example_batch, target_lead_times=target_lead_times, **dataclasses.asdict(task_config) )

eval_inputs, eval_targets, eval_forcings = \
	data_utils.extract_inputs_targets_forcings( example_batch, target_lead_times=eval_lead_times, **dataclasses.asdict(task_config) )

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

