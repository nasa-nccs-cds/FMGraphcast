from fmbase.util.config import cfg
from fmbase.util.ops import fmbdir
from graphcast import checkpoint
from typing import Any, Mapping, Sequence, Tuple, Union, Dict
from graphcast.graphcast import ModelConfig, TaskConfig, CheckPoint
from fmgraphcast.data_utils import load_merra2_norm_data
import xarray as xa

def config_model( **kwargs ) -> ModelConfig:
	opts = dict(
		resolution=     kwargs.get( 'res',           cfg().model.res      ),
		mesh_size=      kwargs.get( 'mesh_size',     cfg().model.mesh_size       ),
		latent_size=    kwargs.get( 'latent_size',   cfg().model.latent_size     ),
		gnn_msg_steps=  kwargs.get( 'gnn_msg_steps', cfg().model.gnn_msg_steps   ),
		hidden_layers=  kwargs.get( 'hidden_layers', cfg().model.hidden_layers   ),
		radius_query_fraction_edge_length= kwargs.get( 'radius_query_fraction_edge_length', cfg().model.radius_query_fraction_edge_length ) )
	ctypes = {k: type(v) for k, v in opts.items()}
	return ModelConfig(**opts)

def config_task( **kwargs) -> TaskConfig:
	dts = cfg().task.data_timestep
	opts = dict(
	    input_variables=    kwargs.get('input_variables',    list(cfg().task.input_variables.keys())),
	    target_variables=   kwargs.get('target_variables',   list(cfg().task.target_variables)),
	    forcing_variables=  kwargs.get('forcing_variables',  list(cfg().task.forcing_variables)),
	    pressure_levels=    kwargs.get('levels',             list(cfg().task.levels)),
	    input_duration=     kwargs.get('input_duration',     f"{cfg().task.input_steps*dts}h" ) )
	ctypes = { k: (type(v), v) for k,v in opts.items() }
	return TaskConfig(**opts)

def dataset_path(**kwargs) -> str:
	root = fmbdir('model')
	parms = { pid: kwargs.get(pid, cfg().model.get(pid)) for pid in [ 'res', 'levels', 'steps', 'year', 'month', 'day' ] }
	dspath = f"data/era5/res-{parms['res']}_levels-{parms['levels']}_steps-{parms['steps']:0>2}"
	dsfile = f"{parms['year']}-{parms['month']:0>2}-{parms['day']:0>2}.nc"
	return f"{root}/{dspath}/{dsfile}"

def load_era5_params() -> Tuple[Dict,Dict]:
	root = fmbdir('model')
	params_file = cfg().task.params
	pfilepath = f"{root}/params/{params_file}.npz"
	with open(pfilepath, "rb") as f:
		ckpt = checkpoint.load(f, CheckPoint)
		params = ckpt.params
		state = {}
		return params, state

def config_files(**kwargs) -> Tuple[ModelConfig,TaskConfig]:
	if kwargs.get('checkpoint',False):
		root = fmbdir('model')
		params_file = cfg().task.params
		pfilepath = f"{root}/params/{params_file}.npz"
		with open(pfilepath, "rb") as f:
			ckpt = checkpoint.load(f, CheckPoint)
			mconfig = ckpt.model_config
			tconfig = ckpt.task_config
			print("Model description:\n", ckpt.description, "\n")
			print(f" >> model_config: {mconfig}")
			print(f" >> task_config:  {tconfig}")
			print(f" >> forcing_variables: {[type(fv) for fv in tconfig.forcing_variables]}")
		return mconfig, tconfig
	else:
		return config_model(**kwargs), config_task(**kwargs)

class ModelConfiguration:
	_instance = None
	_instantiated = None

	def __init__(self):
		cvals = config_files()
		self.model_config: ModelConfig = cvals[0]
		self.task_config: TaskConfig = cvals[1]
		self.norm_data: Dict[str, xa.Dataset] = load_merra2_norm_data()

	@classmethod
	def init(cls):
		if cls._instance is None:
			inst = cls()
			cls._instance = inst
			cls._instantiated = cls

	@classmethod
	def instance(cls) -> "ModelConfiguration":
		cls.init()
		return cls._instance

def model_config() -> ModelConfig:
	mc = ModelConfiguration.instance()
	return mc.model_config

def task_config() -> TaskConfig:
	mc = ModelConfiguration.instance()
	return mc.task_config

def norm_data() -> Dict[str, xa.Dataset]:
	mc = ModelConfiguration.instance()
	return mc.norm_data

def cparms()-> Dict:
	mc = ModelConfiguration.instance()
	return dict( model_config=mc.model_config, task_config=mc.task_config, norm_data=mc.norm_data )

