import functools
from typing import Optional, Dict
from fmbase.source.merra2.model import YearMonth, load_batch
from fmgraphcast.data_utils import load_merra2_norm_data
from fmgraphcast.config import config_files
from fmgraphcast.model import train_model
from fmgraphcast import data_utils
import xarray as xa
import hydra, dataclasses, time
from fmbase.util.config import configure, cfg

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-small' )
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

itf = data_utils.extract_inputs_targets_forcings( example_batch, target_lead_times=target_lead_times, **dataclasses.asdict(task_config) )
train_inputs, train_targets, train_forcings = itf

itf = data_utils.extract_inputs_targets_forcings( example_batch, target_lead_times=eval_lead_times, **dataclasses.asdict(task_config) )
eval_inputs, eval_targets, eval_forcings = itf

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


print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

(cmconfig, ctconfig) = config_files( checkpoint=True )
(mconfig, tconfig)   = config_files( checkpoint=False )
for tc in [ ctconfig, tconfig ]:
	iv = [ type(i) for i in tconfig.input_variables ]
	fv = [ type(i) for i in tconfig.forcing_variables ]
	print( f"\n Config types  iv: {iv}")
	print( f"                 fv: {fv}")

train_model( train_inputs, train_targets, train_forcings )


# def with_params(fn):
# 	return functools.partial(fn, params=params, state=state)
#
# run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))
# train_predictions = run_forward_jitted( rng=jax.random.PRNGKey(0), inputs=train_inputs, targets_template=train_targets * np.nan, forcings=train_forcings)
# eval_predictions = rollout.chunked_prediction( run_forward_jitted, rng=jax.random.PRNGKey(0), inputs=eval_inputs, targets_template=eval_targets * np.nan, forcings=eval_forcings)
#
#
#

# print( f" ***** Completed forecast, result variables:  ")
# for vname, dvar in predictions.data_vars.items():
# 	print( f" > {vname}{dvar.dims}: {dvar.shape}")
# 	ndvar: np.ndarray = dvar.values
# 	tvar: Optional[xa.DataArray] = dvar.coords.get('time')
# 	print(f"   --> dtype: {dvar.dtype}, range: ({ndvar.min():.3f},{ndvar.max():.3f}), mean,std: ({ndvar.mean():.3f},{ndvar.std():.3f}), time: {format_timedeltas(tvar)}")
#
# t1 = time.time()
# print( f"Completed forecast in {t1-t0} sec.")
#



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


