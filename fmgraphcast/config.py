from fmbase.util.config import cfg
from fmbase.util.ops import fmbdir
from graphcast import checkpoint
from typing import Any, Mapping, Sequence, Tuple, Union, Dict
from graphcast.graphcast import ModelConfig, TaskConfig, CheckPoint

def config_model( **kwargs ) -> ModelConfig:
	opts = dict(
		resolution=     kwargs.get( 'resolution',    cfg().model.resolution      ),
		mesh_size=      kwargs.get( 'mesh_size',     cfg().model.mesh_size       ),
		latent_size=    kwargs.get( 'latent_size',   cfg().model.latent_size     ),
		gnn_msg_steps=  kwargs.get( 'gnn_msg_steps', cfg().model.gnn_msg_steps   ),
		hidden_layers=  kwargs.get( 'hidden_layers', cfg().model.hidden_layers   ),
		radius_query_fraction_edge_length= kwargs.get( 'radius_query_fraction_edge_length', cfg().model.radius_query_fraction_edge_length ) )
	return ModelConfig(**opts)

def config_task( **kwargs) -> TaskConfig:
	dts = cfg().task.data_timestep
	opts = dict(
	    input_variables=    kwargs.get('input_variables',    cfg().task.input_variables),
	    target_variables=   kwargs.get('target_variables',   cfg().task.target_variables),
	    forcing_variables=  kwargs.get('forcing_variables',  cfg().task.forcing_variables),
	    pressure_levels=    kwargs.get('levels',             cfg().task.levels),
	    input_duration=     kwargs.get('input_duration',     f"{cfg().task.input_steps*dts}h" ) )
	return TaskConfig(**opts)

def dataset_path(**kwargs) -> str:
	root = fmbdir('model')
	parms = { pid: kwargs.get(pid, cfg().model.get(pid)) for pid in [ 'res', 'levels', 'steps', 'year', 'month', 'day' ] }
	dspath = f"data/era5/res-{parms['res']}_levels-{parms['levels']}_steps-{parms['steps']:0>2}"
	dsfile = f"{parms['year']}-{parms['month']:0>2}-{parms['day']:0>2}.nc"
	return f"{root}/{dspath}/{dsfile}"

def config_files() -> Tuple[ModelConfig,TaskConfig]:
	root = fmbdir('model')
	params_file = cfg().task.params
	pfilepath = f"{root}/params/{params_file}.npz"
	with open(pfilepath, "rb") as f:
		ckpt = checkpoint.load(f, CheckPoint)
		model_config = ckpt.model_config
		task_config = ckpt.task_config
		print("Model description:\n", ckpt.description, "\n")
		print(f" >> model_config: {model_config}")
		print(f" >> task_config:  {task_config}")
	return model_config, task_config

