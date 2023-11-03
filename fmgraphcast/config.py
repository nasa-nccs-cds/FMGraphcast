from graphcast.graphcast import ModelConfig, TaskConfig, GraphCast
from pathlib import Path
def config_model( **kwargs ) -> ModelConfig:
	opts = dict(
		resolution=     kwargs.get('resolution',    cfg().model.resolution),
		mesh_size=      kwargs.get('mesh_size',     cfg().model.mesh_size),
		latent_size=    kwargs.get('latent_size',   cfg().model.latent_size),
		gnn_msg_steps=  kwargs.get('gnn_msg_steps', cfg().model.gnn_msg_steps),
		hidden_layers=  kwargs.get('hidden_layers', cfg().model.hidden_layers),
		radius_query_fraction_edge_length= kwargs.get('radius_query_fraction_edge_length', cfg().model.radius_query_fraction_edge_length) )
	return ModelConfig(**opts)

def config_task( **kwargs) -> TaskConfig:
	dts = cfg().task.data_timestep
	opts = dict(
	    input_variables=    kwargs.get('input_variables',    cfg().task.input_variables),
	    target_variables=   kwargs.get('target_variables',   cfg().task.target_variables),
	    forcing_variables=  kwargs.get('forcing_variables',  cfg().task.forcing_variables),
	    pressure_levels=    kwargs.get('z_levels',           cfg().task.z_levels),
	    input_duration=     kwargs.get('input_duration',     f"{cfg().task.input_steps*dts}h" ) )
	return TaskConfig(**opts)

import hydra, os
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

def cfg() -> DictConfig:
    return Configuration.instance().cfg

def configure(config_name: str, config_path=None):
    if config_path is None: config_path =  "../config"
    Configuration.init( config_name, config_path )

class Configuration:
    _instance = None
    _instantiated = None

    def __init__(self, config_name: str, config_path: str ):
        hydra.initialize( version_base=None, config_path=config_path )
        self.cfg: DictConfig = hydra.compose( config_name, return_hydra_config=True )

    @classmethod
    def init(cls, config_name: str, config_path: str ):
        if cls._instance is None:
            inst = cls(config_name,config_path)
            cls._instance = inst
            cls._instantiated = cls

    @classmethod
    def instance(cls) -> "Configuration":
        return cls._instance