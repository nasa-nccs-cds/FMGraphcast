import traceback

from fmbase.source.merra2.model import FMBatch, BatchType
from fmgraphcast.config import save_params, load_params
from fmgraphcast.model import run_forward, loss_fn, grads_fn, drop_state
import xarray as xa
import functools
from fmgraphcast import data_utils
from graphcast import model_utils
import jax, time
import numpy as np
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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load MERRA2 Data
#-----------------

dts         = cfg().task.data_timestep
target_lead_times = [ f"{iS*dts}h" for iS in range(1,train_steps+1) ]
eval_lead_times =   [ f"{iS*dts}h" for iS in range(1,eval_steps+1) ]
train_dates = year_range( *cfg().task.year_range )
forecast_date = date( cfg().task.year, cfg().task.month, cfg().task.day )
batch_days = cfg().task.input_steps + cfg().task.train_steps

fmbatch: FMBatch = FMBatch( cfg().task, BatchType.Training )
norms: Dict[str, xa.Dataset] = fmbatch.norm_data

def with_configs(fn): return functools.partial( fn, model_config=model_config, task_config=task_config, norms=norms)
def with_params(fn): return functools.partial(fn, params=params, state=state)
init_jitted = jax.jit(with_configs(run_forward.init))
grads_fn_jitted = jax.jit(with_configs(grads_fn))

example_batch: xa.Dataset = fmbatch.load_batch( date_list(forecast_date,batch_days) )
itf = data_utils.extract_inputs_targets_forcings( example_batch, target_lead_times=target_lead_times, **dataclasses.asdict(task_config) )
train_inputs, train_targets, train_forcings = itf

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

stacked_inputs: xa.DataArray = model_utils.dataset_to_stacked( train_inputs ).squeeze()
ndvar: np.ndarray = stacked_inputs.values
lat, lon = stacked_inputs.coords['lat'].values, stacked_inputs.coords['lon'].values
phi = np.pi/180.0
latf, lonf = np.cos( lat*phi ), np.sin( lon*phi )
print( f"\n ** STACKED INPUTS {stacked_inputs.dims}: shape: {stacked_inputs.shape}, dtype: {stacked_inputs.dtype}, range: ({ndvar.min():.3f},{ndvar.max():.3f}), mean,std: ({ndvar.mean():.3f},{ndvar.std():.3f})")
print( f"\n ---------- lat{lat.shape}:  {lat}"  )
print( f"\n ---------- latf{latf.shape}: {[f'{v:.2f}' for v in latf]}" )
print( f"\n ---------- lon{lon.shape}:  {lon}"  )
print( f"\n ---------- lonf{lonf.shape}: {[f'{v:.2f}' for v in lonf]}" )

loss, diagnostics, next_state, grads = with_params(grads_fn_jitted)(inputs=train_inputs, targets=train_targets, forcings=train_forcings)




