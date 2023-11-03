from graphcast.graphcast import ModelConfig, TaskConfig, GraphCast
from fmbase.util.config import cfg
def config_model( **kwargs ) -> ModelConfig:
	mc = ModelConfig()
	mc.resolution=     kwargs.get('resolution',    cfg().model.resolution),
	mc.mesh_size=      kwargs.get('mesh_size',     cfg().model.mesh_size),
	mc.latent_size=    kwargs.get('latent_size',   cfg().model.latent_size),
	mc.gnn_msg_steps=  kwargs.get('gnn_msg_steps', cfg().model.gnn_msg_steps),
	mc.hidden_layers=  kwargs.get('hidden_layers', cfg().model.hidden_layers),
	mc.radius_query_fraction_edge_length= kwargs.get('radius_query_fraction_edge_length', cfg().model.radius_query_fraction_edge_length)
	return mc

def config_task( **kwargs) -> TaskConfig:
	dts = cfg().task.data_timestep
	tc =  TaskConfig()
	tc.input_variables=    kwargs.get('input_variables',    cfg().task.input_variables)
	tc.target_variables=   kwargs.get('target_variables',   cfg().task.target_variables)
	tc.forcing_variables=  kwargs.get('forcing_variables',  cfg().task.forcing_variables)
	tc.pressure_levels=    kwargs.get('z_levels',           cfg().task.z_levels)
	tc.input_duration=     kwargs.get('input_duration',     f"{cfg().task.input_steps*dts}h" )
	return tc