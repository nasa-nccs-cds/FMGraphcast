from graphcast import data_utils
import dataclasses
from graphcast.graphcast import ModelConfig, TaskConfig, GraphCast
from fmgraphcast.config import config_model, config_task, cfg
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmbase.source.merra2.model import MERRA2DataInterface, YearMonth
from fmbase.util.config import configure
import functools, xarray as xa
configure( 'explore-test1' )

params = None
state = {}
mconfig: ModelConfig = config_model()
tconfig: TaskConfig = config_task()
train_steps = cfg().task.train_steps
input_steps = cfg().task.input_steps
eval_steps  = cfg().task.eval_steps
dts         = cfg().task.data_timestep
start = YearMonth(2000,0)
end = YearMonth(2000,1)

datasetMgr = MERRA2DataInterface()
example_batch: xa.Dataset = datasetMgr.load_batch(start,end)

print("Loaded Batch:")
for vname, dvar in example_batch.data_vars.items():
	print( f" {vname}{list(dvar.dims)}: shape={dvar.shape}")

norm_data: Dict[str,xa.Dataset] = datasetMgr.load_norm_data()

print("Loaded Norm Data:")
for vname, ndset in norm_data.items():
	print( f" {vname}.norm: shape={ndset.data_vars['norm'].shape}")
	print( f" {vname}.std:  shape={ndset.data_vars['std'].shape}")

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice(f"{dts}h", f"{train_steps*dts}h"), **dataclasses.asdict(tconfig) )

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice(f"{dts}h", f"{eval_steps*dts}h"), **dataclasses.asdict(tconfig) )

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)