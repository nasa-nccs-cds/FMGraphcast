import fmbase.util.data as data_utils
import hydra, dataclasses
from fmbase.util.config import configure, cfg
from graphcast.graphcast import ModelConfig, TaskConfig, GraphCast
from fmgraphcast.config import config_model, config_task
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmbase.source.merra2.model import YearMonth, load_batch
from fmbase.source.merra2.preprocess import load_norm_data
import functools, xarray as xa
hydra.initialize( version_base=None, config_path="../config" )
configure( 'explore-test1' )

params = None
state = {}
mconfig: ModelConfig = config_model()
tconfig: TaskConfig = config_task()
train_steps = cfg().task.train_steps
input_steps = cfg().task.input_steps
eval_steps  = cfg().task.eval_steps
dts         = cfg().task.data_timestep
coords = dict( z="level" )
start = YearMonth(2000,0)
end = YearMonth(2000,1)
target_lead_times = [ f"{iS*dts}h" for iS in range(1,train_steps+1) ]
eval_lead_times =   [ f"{iS*dts}h" for iS in range(1,eval_steps+1) ]

example_batch: xa.Dataset = load_batch( start, end, coords=coords )

print("Loaded Batch:")
for vname, dvar in example_batch.data_vars.items():
	print( f" {vname}{list(dvar.dims)}: shape={dvar.shape}")

norm_data: Dict[str,xa.Dataset] = load_norm_data()

print("Loaded Norm Data:")
for vname, ndset in norm_data.items():
	for nname, ndata in ndset.data_vars.items():
		print( f" {vname}.{nname}: shape={ndata.shape}")

train_inputs, train_targets, train_forcings = \
	data_utils.extract_inputs_targets_forcings( example_batch, target_lead_times=target_lead_times, **dataclasses.asdict(tconfig) )

eval_inputs, eval_targets, eval_forcings = \
	data_utils.extract_inputs_targets_forcings( example_batch, target_lead_times=eval_lead_times, **dataclasses.asdict(tconfig) )

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)