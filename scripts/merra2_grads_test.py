import traceback
from fmbase.source.merra2.model import FMBatch, BatchType
from fmgraphcast.config import save_params, load_params
from fmgraphcast.model import run_forward, loss_fn, grads_fn, drop_state
from fmbase.util.logging import lgm, exception_handled, log_timing
import xarray as xa
import functools
from fmgraphcast import data_utils
import jax, time
import numpy as np
import random
import hydra, dataclasses
from datetime import date
from fmbase.util.dates import date_list, year_range
from fmbase.util.config import configure, cfg
from typing import List, Union, Tuple, Optional, Dict, Type

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-finetuning' )
t0 = time.time()

def parse_file_parts(file_name):
	return dict(part.split("-", 1) for part in file_name.split("_"))

def dtypes( d: Dict ):
	return { k: type(v) for k,v in d.items() }

year, month, day =  cfg().task.year,  cfg().task.month,  cfg().task.day
train_steps = cfg().task.train_steps
runid = "small"
(params, model_config, task_config) = load_params("merra2", runid=runid, hydra_config=False )
state = {}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load MERRA2 Data
#-----------------

dts         = cfg().task.data_timestep
target_lead_times = [ f"{iS*dts}h" for iS in range(1,train_steps+1) ]
fmbatch: FMBatch = FMBatch( cfg().task, BatchType.Training )
norms: Dict[str, xa.Dataset] = fmbatch.norm_data
day_offset = 0
forecast_date= date( cfg().task.year, cfg().task.month, cfg().task.day )

def with_configs(fn): return functools.partial( fn, model_config=model_config, task_config=task_config, norms=norms)
def with_params(fn): return functools.partial(fn, params=params, state=state)
init_jitted = jax.jit(with_configs(run_forward.init))
grads_fn_jitted = jax.jit(with_configs(grads_fn))

print( f"Forecast date: {forecast_date}")
fmbatch.load_batch( forecast_date )
train_data: xa.Dataset = fmbatch.get_train_data( day_offset )
itf = data_utils.extract_inputs_targets_forcings( train_data, target_lead_times=target_lead_times, **dataclasses.asdict(task_config) )
train_inputs, train_targets, train_forcings = itf

loss, diagnostics, next_state, grads = with_params(grads_fn_jitted)( inputs=train_inputs, targets=train_targets, forcings=train_forcings )
print(f" ** loss {loss:.4f}, diagnostics: {type(diagnostics)}, next_state: {type(next_state)}")
