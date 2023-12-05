from fmbase.util.config import cfg
from fmbase.util.ops import fmbdir
from graphcast import checkpoint
from typing import Any, Mapping, Sequence, Tuple, Union, Dict, Optional
from graphcast.graphcast import ModelConfig, TaskConfig, CheckPoint
from fmgraphcast.data_utils import load_merra2_norm_data
import xarray as xa, os, chex

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


@chex.dataclass(frozen=True, eq=True)
class FMCheckPoint:
	params: dict[str, Any]
	model_config: ModelConfig
	task_config: TaskConfig

def cpfilepath(**kwargs) -> str:
	runid = kwargs.get('runid', 'default')
	pdir = f"{fmbdir('results')}/params"
	os.makedirs( pdir, mode=0o777, exist_ok=True )
	params_file =  f"{cfg().task.dataset_version}.{runid}.npz"
	return f"{pdir}/{params_file}"

def load_merra2_params(**kwargs) -> Tuple[Dict,ModelConfig,TaskConfig]:
	pfile = cpfilepath(**kwargs)
	with open(pfile, "rb") as f:
		ckpt: FMCheckPoint = checkpoint.load(f, FMCheckPoint)
		print(f" Loading merra2 model weights from file: {pfile}")
		return ckpt.params, ckpt.model_config, ckpt.task_config

def save_params( params: Dict, modelconfig: ModelConfig, taskconfig: TaskConfig, **kwargs ):
	pfile = cpfilepath(**kwargs)
	with open(pfile,"wb") as f:
		ckpt: FMCheckPoint = FMCheckPoint( params=params, model_config=modelconfig, task_config=taskconfig )
		checkpoint.dump( f, ckpt )
		print( f" Saving model weights to file: {pfile}")

def load_era5_params() -> Tuple[Dict,ModelConfig,TaskConfig]:
	from graphcast.graphcast import CheckPoint
	root = fmbdir('model')
	params_file = cfg().task.params
	pfile = f"{root}/params/{params_file}.npz"
	with open(pfile, "rb") as f:
		ckpt: CheckPoint = checkpoint.load(f, CheckPoint)
		print(f" Loading era5 model weights from file: {pfile}")
		return ckpt.params, ckpt.model_config, ckpt.task_config

def load_params( ptype: str, **kwargs ) -> Optional[Tuple[Dict,ModelConfig,TaskConfig]]:
	(hy_mconfig, hy_tconfig) = hydra_config_files()
	use_hydra = kwargs.get( "hydra_config", False )
	if ptype.startswith("era"):
		params, mconfig, tconfig = load_era5_params()
		if use_hydra: mconfig, tconfig = hy_mconfig, hy_tconfig
	elif ptype.startswith("merra"):
		params, mconfig, tconfig =  load_merra2_params(**kwargs)
		if use_hydra: mconfig, tconfig = hy_mconfig, hy_tconfig
	else:
		params, mconfig, tconfig = None, hy_mconfig, hy_tconfig
	return params, mconfig, tconfig

def hydra_config_files(**kwargs) -> Tuple[ModelConfig,TaskConfig]:
	return config_model(**kwargs), config_task(**kwargs)

class ModelConfiguration:
	_instance = None
	_instantiated = None

	def __init__(self):
		cvals = hydra_config_files()
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

