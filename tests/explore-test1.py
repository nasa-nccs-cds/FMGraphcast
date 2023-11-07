from graphcast.predictor_base import Predictor
from graphcast import autoregressive
from graphcast import casting
from graphcast import graphcast
from graphcast import normalization
from graphcast import xarray_jax
from graphcast import xarray_tree
import jax, haiku as hk
import fmbase.util.data as data_utils
import hydra, dataclasses
from graphcast.graphcast import ModelConfig, TaskConfig, GraphCast
from fmgraphcast.config import config_model, config_task
from fmbase.util.config import configure, cfg
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmbase.source.merra2.model import MERRA2DataInterface, YearMonth
import functools, xarray as xa
hydra.initialize( version_base=None, config_path="../config" )
configure( 'explore-test1' )

params = None
state = {}
mconfig: ModelConfig = config_model()
tconfig: TaskConfig = config_task()
train_steps = cfg().task.train_steps
input_steps = cfg().task.input_steps
eval_steps  = cfg().task.eval_steps
dts         = cfg().task.data_timestep
coords = dict( z="level" )
start = YearMonth(2000,0)
end = YearMonth(2000,1)
target_lead_times = [ f"{iS*dts}h" for iS in range(1,train_steps+1) ]
eval_lead_times =   [ f"{iS*dts}h" for iS in range(1,eval_steps+1) ]

datasetMgr = MERRA2DataInterface()
example_batch: xa.Dataset = datasetMgr.load_batch( start, end, coords=coords )

print("Loaded Batch:")
for vname, dvar in example_batch.data_vars.items():
	print( f" {vname}{list(dvar.dims)}: shape={dvar.shape}")

norm_data: Dict[str,xa.Dataset] = datasetMgr.load_norm_data()

print("Loaded Norm Data:")
for vname, ndset in norm_data.items():
	for nname, ndata in ndset.data_vars.items():
		print( f" {vname}.{nname}: shape={ndata.shape}")

train_inputs, train_targets, train_forcings = \
	data_utils.extract_inputs_targets_forcings( example_batch, target_lead_times=target_lead_times, **dataclasses.asdict(tconfig) )

eval_inputs, eval_targets, eval_forcings = \
	data_utils.extract_inputs_targets_forcings( example_batch, target_lead_times=eval_lead_times, **dataclasses.asdict(tconfig) )

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)

def construct_wrapped_graphcast( model_config: ModelConfig, task_config: TaskConfig, **kwargs ) -> Predictor:
	gcast = graphcast.GraphCast(model_config, task_config)
	predictor: Predictor = casting.Bfloat16Cast(gcast)
#           Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from BFloat16 happens after applying normalization to the inputs/targets.
	npredictor = normalization.InputsAndResiduals( predictor, mean_by_level=norm_data['mean'], stddev_by_level=norm_data['std'], diffs_stddev_by_level=norm_data['std'])  # TODO: diffs_stddev_by_level=norm_data['std_diff']
#           Wraps everything so the one-step model can produce trajectories.
	apredictor = autoregressive.Predictor(npredictor, gradient_checkpointing=True)
	return apredictor

@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)

@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure( lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),(loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
     (loss, diagnostics), next_state = loss_fn.apply( params, state, jax.random.PRNGKey(0), model_config, task_config, i, t, f)
     return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad( _aux, has_aux=True )(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial( fn, model_config=mconfig, task_config=tconfig)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit( with_configs( run_forward.init ) )

if params is None:
  params, state = init_jitted( rng=jax.random.PRNGKey(0), inputs=train_inputs, targets_template=train_targets, forcings=train_forcings )

loss_fn_jitted = drop_state( with_params( jax.jit( with_configs( loss_fn.apply ) ) ) )
grads_fn_jitted = with_params( jax.jit( with_configs( grads_fn ) ) )
run_forward_jitted = drop_state( with_params( jax.jit( with_configs( run_forward.apply ) ) ) )

