from fmgraphcast import data_utils
from fmgraphcast.config import config_files
from fmgraphcast.config import dataset_path
import xarray as xa
import hydra, dataclasses
from fmbase.util.config import configure, cfg

hydra.initialize( version_base=None, config_path="../config" )
configure( 'explore-era5' )

def parse_file_parts(file_name):
	return dict(part.split("-", 1) for part in file_name.split("_"))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load  ERA5  Data
#-----------------

print( "  --------------------- ERA5 ---------------------")
(model_config,task_config) = config_files()
dataset_file = dataset_path(year=2022)
with open(dataset_file,"rb") as f:
	example_batch: xa.Dataset = xa.load_dataset(f).compute()
print( "Batch Time coord:  ", example_batch.coords['time'] )

# Extract training and eval data

train_steps, eval_steps = cfg().task.train_steps, cfg().task.eval_steps
train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings( example_batch,
												target_lead_times=slice("6h", f"{train_steps*6}h"), **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings( example_batch,
												target_lead_times=slice("6h", f"{eval_steps*6}h"), **dataclasses.asdict(task_config))

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
