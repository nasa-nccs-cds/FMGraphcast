import functools
from typing import Optional, Dict
from fmbase.source.merra2.model import YearMonth, load_batch
from fmgraphcast.data_utils import load_merra2_norm_data
from fmgraphcast.config import config_files
from fmgraphcast.model import run_forward, drop_state, grads_fn, loss_fn
from fmbase.util.ops import format_timedeltas, print_dict
from fmgraphcast import data_utils
from graphcast import rollout
import haiku as hk
import jax
import numpy as np
import xarray as xa
import hydra, dataclasses, time
from fmbase.util.config import configure, cfg

hydra.initialize( version_base=None, config_path="../config" )
configure( 'explore-era5' )
t0 = time.time()

def parse_file_parts(file_name):
	return dict(part.split("-", 1) for part in file_name.split("_"))

def dtypes( d: Dict ):
	return { k: type(v) for k,v in d.items() }

res,levels,steps = cfg().model.res,  cfg().model.levels,  cfg().model.steps
year, month, day =  cfg().model.year,  cfg().model.month,  cfg().model.day
train_steps, eval_steps = cfg().task.train_steps, cfg().task.eval_steps
(model_config,task_config) = config_files()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load MERRA2 Data
#-----------------

params: Dict[str,Dict] = None
state = {}
dts         = cfg().task.data_timestep
start = YearMonth(year,month)
end = YearMonth(year,month+1)
target_lead_times = [ f"{iS*dts}h" for iS in range(1,train_steps+1) ]
eval_lead_times =   [ f"{iS*dts}h" for iS in range(1,eval_steps+1) ]

print( "  --------------------- MERRA2 ---------------------")
example_batch: xa.Dataset = load_batch( start, end, cfg().task )

print("Loaded Batch:")
for vname, dvar in example_batch.data_vars.items():
	print( f" {vname}{list(dvar.dims)}: shape={dvar.shape}")

norm_data: Dict[str,xa.Dataset] = load_merra2_norm_data()

# print("\n Loaded Norm Data:")
# for vname, ndset in norm_data.items():
# 	print( f"------------ Norm dataset: {vname} ------------ " )
# 	for nname, ndata in ndset.data_vars.items():
# 		print( f"   ** {vname}.{nname}: shape={ndata.shape}")

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

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
	return functools.partial( fn, model_config=model_config, task_config=task_config, norm_data=norm_data )

# Always pass params and state, so the usage below are simpler
def with_params(fn):
	return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
	return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
	params, state = init_jitted( rng=jax.random.PRNGKey(0), inputs=train_inputs, targets_template=train_targets, forcings=train_forcings)

loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

# Autoregressive rollout (loop in python)

print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

predictions: xa.Dataset = rollout.chunked_prediction( run_forward_jitted, rng=jax.random.PRNGKey(0), inputs=eval_inputs,
														        targets_template=eval_targets * np.nan, forcings=eval_forcings)

print( f" ***** Completed forecast, result variables:  ")
for vname, dvar in predictions.data_vars.items():
	print( f" > {vname}{dvar.dims}: {dvar.shape}")
	ndvar: np.ndarray = dvar.values
	tvar: Optional[xa.DataArray] = dvar.coords.get('time')
	print(f"   --> dtype: {dvar.dtype}, range: ({ndvar.min():.3f},{ndvar.max():.3f}), mean,std: ({ndvar.mean():.3f},{ndvar.std():.3f}), time: {format_timedeltas(tvar)}")

t1 = time.time()
print( f"Completed forecast in {t1-t0} sec.")


#  Loss computation (autoregressive loss over multiple steps)

loss0, diagnostics0 = loss_fn_jitted( rng=jax.random.PRNGKey(0), inputs=train_inputs, targets=train_targets, forcings=train_forcings)
print("Loss:", float(loss0))

#  Gradient computation (backprop through time)

loss1, diagnostics1, next_state, grads = grads_fn_jitted( inputs=train_inputs, targets=train_targets, forcings=train_forcings)
mean_grad = np.mean( jax.tree_util.tree_flatten( jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads) )[0] )

print("\n----------------------------------------------------")
print( f"Params:")
for k,vdict in params.items():
	print( f" ** {k}:\t\t{dtypes(vdict)}")
print( f"Grads:")
for k,vdict in grads.items():
	print( f" ** {k}:\t\t{dtypes(vdict)}")
print(f"Loss: {loss1:.4f}, Mean |grad|: {mean_grad:.6f}")

# Autoregressive rollout (keep the loop in JAX)

print("Inputs:  ", train_inputs.dims.mapping)
print("Targets: ", train_targets.dims.mapping)
print("Forcings:", train_forcings.dims.mapping)

predictions = run_forward_jitted( rng=jax.random.PRNGKey(0), inputs=train_inputs, targets_template=train_targets * np.nan, forcings=train_forcings)

print( f"Completed Autoregressive rollout in {(time.time()-t1)/60} min.")


#
# @jit
# def update(params, batch):
# 	grads = grad(loss)(params, batch)
# 	return [(w - step_size * dw, b - step_size * db)
# 		for (w, b), (dw, db) in zip(params, grads)]
#
# params = init_random_params(param_scale, layer_sizes)
# for epoch in range(num_epochs):
# 	start_time = time.time()
# 	for _ in range(num_batches):
# 		params = update(params, next(batches))
# 	epoch_time = time.time() - start_time
#
# 	train_acc = accuracy(params, (train_images, train_labels))
# 	test_acc = accuracy(params, (test_images, test_labels))
# 	print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
# 	print(f"Training set accuracy {train_acc}")
# 	print(f"Test set accuracy {test_acc}")
#


