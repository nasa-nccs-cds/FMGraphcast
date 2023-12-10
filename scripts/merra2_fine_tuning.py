from fmbase.source.merra2.model import FMBatch
from fmgraphcast.config import save_params, load_params
from fmgraphcast.model import run_forward, loss_fn, grads_fn, drop_state
import xarray as xa
import functools
from graphcast import autoregressive
from graphcast import casting
from fmgraphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax, time
import numpy as np
import hydra, dataclasses
from datetime import date
from fmbase.util.dates import date_list
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
def with_configs(fn): return functools.partial( fn, model_config=model_config, task_config=task_config)
def with_params(fn): return functools.partial(fn, params=params, state=state)
init_jitted = jax.jit(with_configs(run_forward.init))
grads_fn_jitted = jax.jit(with_configs(grads_fn))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load MERRA2 Data
#-----------------

dts         = cfg().task.data_timestep
start = date(1990,1,1 ) # date( cfg().task.year, cfg().task.month, cfg().task.day )
ndays = 3
target_lead_times = [ f"{iS*dts}h" for iS in range(1,train_steps+1) ]
eval_lead_times =   [ f"{iS*dts}h" for iS in range(1,eval_steps+1) ]
train_dates = date_list( start, 10 )
nepochs = 2 # cfg().task.nepochs
niter = 20

fmbatch = FMBatch( cfg().task )

for epoch in range(nepochs):
	print(f"\n -------------------------------- Epoch {epoch} -------------------------------- \n")
	for forecast_date in train_dates:
		example_batch: xa.Dataset = fmbatch.load_batch( date_list(forecast_date,ndays) )
		itf = data_utils.extract_inputs_targets_forcings( example_batch, target_lead_times=target_lead_times, **dataclasses.asdict(task_config) )
		train_inputs, train_targets, train_forcings = itf

		itf = data_utils.extract_inputs_targets_forcings( example_batch, target_lead_times=eval_lead_times, **dataclasses.asdict(task_config) )
		eval_inputs, eval_targets, eval_forcings = itf

		if params is None:
			params, state = init_jitted( rng=jax.random.PRNGKey(0), inputs=train_inputs, targets_template=train_targets, forcings=train_forcings)

		loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
		run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

		for iteration in range(niter):
			te= time.time()
			loss, diagnostics, next_state, grads = with_params(grads_fn_jitted)( inputs=train_inputs, targets=train_targets, forcings=train_forcings )
			mean_grad = np.mean( jax.tree_util.tree_flatten( jax.tree_util.tree_map( lambda x: np.abs(x).mean(), grads ) )[0] )
			max_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).max(), grads))[0])
			params = jax.tree_map(  lambda p, g: p - lr * g, params, grads)
			print(f" * ITER {epoch}:{iteration}: Loss= {loss:.6f}, Mean/Max |dW|= {lr*mean_grad:.6f} / {lr*max_grad:.6f}, comptime= {time.time()-te:.1f} sec")

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