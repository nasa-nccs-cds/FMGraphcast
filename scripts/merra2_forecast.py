import traceback
from fmbase.source.merra2.model import FMBatch
from fmgraphcast.config import save_params, load_params
from fmgraphcast.model import run_forward, loss_fn, grads_fn, drop_state
import xarray as xa
import functools
from graphcast import rollout
import jax, time
import numpy as np
from fmgraphcast import data_utils
import hydra, dataclasses
from datetime import date
from fmbase.util.ops import format_timedeltas
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
forecast_steps = 4
runid = "small"
(params, model_config, task_config) = load_params("merra2", runid=runid, hydra_config=False )
state = {}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load MERRA2 Data
#-----------------

dts         = cfg().task.data_timestep
eval_lead_times =   [ f"{iS*dts}h" for iS in range(1,forecast_steps+1) ]
fmbatch: FMBatch = FMBatch( cfg().task )
norms: Dict[str, xa.Dataset] = fmbatch.norm_data
reference_date = date( year, month, day )
ref_day_offset = 0

def with_configs(fn): return functools.partial( fn, model_config=model_config, task_config=task_config, norms=norms)
def with_params(fn): return functools.partial(fn, params=params, state=state)

run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))
fmbatch.load_batch(reference_date)

eval_data: xa.Dataset = fmbatch.get_train_data(ref_day_offset)
itf = data_utils.extract_inputs_targets_forcings(eval_data, target_lead_times=eval_lead_times, **dataclasses.asdict(task_config))
eval_inputs, eval_targets, eval_forcings = itf

predictions: xa.Dataset = rollout.chunked_prediction( run_forward_jitted, rng=jax.random.PRNGKey(0), inputs=eval_inputs,
															targets_template=eval_targets * np.nan, forcings=eval_forcings)

print( f" ***** Completed forecast, result variables:  ")
for vname, dvar in predictions.data_vars.items():
	print( f" > {vname}{dvar.dims}: {dvar.shape}")
	ndvar: np.ndarray = dvar.values
	tvar: Optional[xa.DataArray] = dvar.coords.get('time')
	print(f"   --> dtype: {dvar.dtype}, range: ({ndvar.min():.3f},{ndvar.max():.3f}), mean,std: ({ndvar.mean():.3f},{ndvar.std():.3f}), time: {format_timedeltas(tvar)}")


print( f"Completed in {time.time()-t0} sec.")


