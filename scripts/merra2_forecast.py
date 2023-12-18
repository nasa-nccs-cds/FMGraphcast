import traceback
from fmbase.source.merra2.model import FMBatch, BatchType
from fmgraphcast.config import save_params, load_params
from fmgraphcast.model import run_forward, loss_fn, grads_fn, drop_state
import xarray as xa
import functools
from fmgraphcast import data_utils
from fmbase.util.ops import format_timedeltas
from graphcast import rollout
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

res,levels= cfg().model.res,  cfg().task.levels
year, month, day =  cfg().task.year,  cfg().task.month,  cfg().task.day
train_steps, eval_steps = cfg().task.train_steps, cfg().task.eval_steps
runid = "small"
(params, model_config, task_config) = load_params("merra2", runid=runid, hydra_config=False )
state = {}
lr = cfg().task.lr
reference_date = date( year, month, day )
day_offset = 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load MERRA2 Data
#-----------------

dts         = cfg().task.data_timestep
target_leadtimes = [ f"{iS*dts}h" for iS in range(1,train_steps+1) ]
eval_leadtimes =   [ f"{iS*dts}h" for iS in range(1,eval_steps+1) ]

fmbatch: FMBatch = FMBatch( cfg().task, BatchType.Forecast )
norms: Dict[str, xa.Dataset] = fmbatch.norm_data
error_threshold = cfg().task.error_threshold
fmbatch.load_batch( reference_date )

def with_configs(fn): return functools.partial( fn, model_config=model_config, task_config=task_config, norms=norms)
def with_params(fn): return functools.partial(fn, params=params, state=state)

init_jitted = jax.jit(with_configs(run_forward.init))
grads_fn_jitted = jax.jit(with_configs(grads_fn))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

train_data: xa.Dataset = fmbatch.get_train_data( day_offset )
itf = data_utils.extract_inputs_targets_forcings( train_data, target_lead_times=eval_leadtimes, **dataclasses.asdict(task_config) )
eval_inputs, eval_targets, eval_forcings = itf

if params is None:
	itf = data_utils.extract_inputs_targets_forcings(train_data, target_lead_times=target_leadtimes, **dataclasses.asdict(task_config))
	train_inputs, train_targets, train_forcings = itf
	params, state = init_jitted( rng=jax.random.PRNGKey(0), inputs=train_inputs, targets_template=train_targets, forcings=train_forcings)

print( f"\nRunning prediction, eval_leadtimes={eval_leadtimes}, template variables:")
for vname, dvar in eval_targets.data_vars.items():
	print( f" > {vname}{dvar.dims}: {dvar.shape}")

ts=time.time()
predictions: xa.Dataset = rollout.chunked_prediction( run_forward_jitted, rng=jax.random.PRNGKey(0), inputs=eval_inputs,
														        targets_template=eval_targets * np.nan, forcings=eval_forcings)

print( f"\n ***** Completed forecast in {time.time()-ts:.3f} sec, result variables:  ")
for vname, dvar in predictions.data_vars.items():
	print( f" > {vname}{dvar.dims}: {dvar.shape}")
	ndvar: np.ndarray = dvar.values
	tvar: Optional[xa.DataArray] = dvar.coords.get('time')
	print(f"   --> dtype: {dvar.dtype}, range: ({ndvar.min():.3f},{ndvar.max():.3f}), mean,std: ({ndvar.mean():.3f},{ndvar.std():.3f}), time: {format_timedeltas(tvar)}")


print( f"Completed in {time.time()-t0} sec.")


