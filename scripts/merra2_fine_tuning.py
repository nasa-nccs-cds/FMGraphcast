import traceback
from fmbase.source.merra2.model import FMBatch
from fmgraphcast.config import save_params, load_params
from fmgraphcast.model import run_forward, loss_fn, grads_fn, drop_state
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

res,levels= cfg().model.res,  cfg().task.levels
year, month, day =  cfg().task.year,  cfg().task.month,  cfg().task.day
train_steps, eval_steps = cfg().task.train_steps, cfg().task.eval_steps
runid = "small"
(params, model_config, task_config) = load_params("merra2", runid=runid, hydra_config=False )
state = {}
lr = cfg().task.lr
output_period = 50

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load MERRA2 Data
#-----------------

dts         = cfg().task.data_timestep
target_lead_times = [ f"{iS*dts}h" for iS in range(1,train_steps+1) ]
eval_lead_times =   [ f"{iS*dts}h" for iS in range(1,eval_steps+1) ]
train_dates = year_range( *cfg().task.year_range, randomize=True )
nepochs = cfg().task.nepoch
niter = cfg().task.niter
fmbatch: FMBatch = FMBatch( cfg().task )
norms: Dict[str, xa.Dataset] = fmbatch.norm_data

def with_configs(fn): return functools.partial( fn, model_config=model_config, task_config=task_config, norms=norms)
def with_params(fn): return functools.partial(fn, params=params, state=state)
init_jitted = jax.jit(with_configs(run_forward.init))
grads_fn_jitted = jax.jit(with_configs(grads_fn))

for epoch in range(nepochs):
	for date_index, forecast_date in enumerate(train_dates):
		print( "\n" + ("\t"*8) + f"EPOCH {epoch} *** Forecast date[{date_index}]: {forecast_date}")
		fmbatch.load_batch( forecast_date )
		losses = {}
		for iteration in range(niter):
			for day_offset in range(0,4):
				train_data: xa.Dataset = fmbatch.get_train_data( day_offset )
				itf = data_utils.extract_inputs_targets_forcings( train_data, target_lead_times=target_lead_times, **dataclasses.asdict(task_config) )
				train_inputs, train_targets, train_forcings = itf
				itf = data_utils.extract_inputs_targets_forcings( train_data, target_lead_times=eval_lead_times, **dataclasses.asdict(task_config) )
				eval_inputs, eval_targets, eval_forcings = itf

				if params is None:
					params, state = init_jitted( rng=jax.random.PRNGKey(0), inputs=train_inputs, targets_template=train_targets, forcings=train_forcings)

				try:
					te= time.time()
					loss, diagnostics, next_state, grads = with_params(grads_fn_jitted)( inputs=train_inputs, targets=train_targets, forcings=train_forcings )
					params = jax.tree_map(  lambda p, g: p - lr * g, params, grads)
					losses.setdefault( day_offset, [] ).append( loss )
				except Exception as err:
					print( f"\n\n ABORT @ {epoch}:{iteration}")
					traceback.print_exc()
					break
		for day_offset in range(0, 4):
			print(f" * LOSS[{day_offset}]= {[f'{L:.3f}' for L in losses[day_offset]]}")

		if date_index % output_period == 0: save_params(params, model_config, task_config, runid=runid)

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
