import jax

from jax import numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils


def my_print(s):
   if yes_print:
      print(s)

def S(*specs):
  return NamedSharding(mesh, PartitionSpec(*specs))

# Initialize shift and state
# shift = jnp.zeros([num_stages] + micro_shape)
# state = jnp.reshape(test_inputs, (num_stages, microbatches // num_stages) + test_inputs.shape[1:])

# Construct stages in
def get_iteration_inputs(loop_iteration, microbatches, num_stages, state, async_state):
    stream_buf_idx = loop_iteration % (microbatches // num_stages)
    stream_slice = state[:,stream_buf_idx] 
    circ_slice = async_state[:,loop_iteration % microbatches]
    return jnp.where(loop_iteration < microbatches, stream_slice, circ_slice) # Circ specialty
    #return state[:,stream_buf_idx] # equivalent to state[:,stream_buf_idx,:,:] # Non-circ

def select_state_or_input(input, state):
    # Selects input for stage 0, state for other stages
    return jnp.where(jax.lax.broadcasted_iota('int32', state.shape, 0) == 0, input, state)

# run model
def get_new_loop_state(output, old_state, old_async_state, old_lir, loop_iteration,n_stages):
    # Rotate state to the right by 1. (for non-circ shift instead of rotate)
    
    # For non-circ
    # def _shift_right(shift_in):
    #     padding = [[1, 0]] + [[0, 0]] * (shift_in.ndim - 1)
    #     # Use lax.slice to guarantee the gradient is a pad.
    #     return jax.lax.slice(jnp.pad(shift_in, padding), [0] * shift_in.ndim, shift_in.shape)
    
    def _rotate_right(output_in):
      # Use lax.slice to avoid generating a gather.
      last = jax.lax.slice_in_dim(output_in, args.n_stages - 1, args.n_stages, axis=0)
      except_last = jax.lax.slice_in_dim(output_in, 0, args.n_stages - 1, axis=0)
      return jnp.concatenate([last, except_last], axis=0)

    # Shift
    new_shift = _rotate_right(output)

    # Async state
    def _rotate_right_and_update(lir_in, async_state_in):
        rotated = _rotate_right(lir_in)
        rotated = jnp.expand_dims(rotated, 1)
        # The offset is the last stage's last microbatch ID.
        #offset = (loop_iteration - (num_stages - 1)) % microbatches # we need extar -1 b/c grabbing from un-updated LIR
        offset = (loop_iteration - (args.n_stages - 1) - 1) % args.n_microbatches # This looks like an extra - 1 to me
        return jax.lax.dynamic_update_slice_in_dim(async_state_in, rotated, offset, axis=1)
    new_async_state = _rotate_right_and_update(old_lir, old_async_state)

    # lir
    new_lir = output


    # Stream state
    stream_buf_idx = loop_iteration % (microbatches // args.n_stages)
    stream_slice = old_state[:, stream_buf_idx]
    def _update_state(state_in, stream_slice, output):
        # Shift the current slice to the left, then fill the last stage with
        # the final output.
        padding = [[0, 1]] + [[0, 0]] * (stream_slice.ndim - 1)
        stream_slice = jax.lax.slice_in_dim(
            jnp.pad(stream_slice, padding), 1, stream_slice.shape[0] + 1, axis=0)
        stream_slice = jnp.where(
            jax.lax.broadcasted_iota('int32', stream_slice.shape, 0) == args.n_stages - 1, output,
            stream_slice)
        stream_slice = jnp.expand_dims(stream_slice, 1)
        return jax.lax.dynamic_update_slice_in_dim(
            state_in, stream_slice, stream_buf_idx, axis=1)
    new_state = _update_state(old_state, stream_slice, output)
    return new_state, new_shift, new_async_state, new_lir

def stage(weights, x):
  #for i in range(weights.shape[0]): # this was used if each stage had multiple layers (we would need to reshape weights to stages, layers/stage, embed, embed)
  x = layer(weights, x)
  return x

def layer(weights, x):
  if sum_layer:
     x_out = weights + x
  else:
    x_out = jnp.einsum('bse,eh->bsh',x,weights) # The leading stage dimension of weights is missing because it is vmapped out
    x_out = jnp.tanh(x_out)

  #x = jnp.tanh(jnp.dot(x, w))
  return x_out

# You are here, does global_layer need to be different for non-pipeline?
def global_layer(weights, x):
    return sad


def get_weights_stage(weights, loop_iteration):
    microbatch_ids = jnp.maximum(loop_iteration - jnp.arange(args.n_stages), 0) # not a great name, really this is like batch_id * repeat idx
    repeat_ids = microbatch_ids // args.n_microbatches
    layer_ids = jnp.arange(args.n_stages) + repeat_ids * args.n_stages
    # layer_idx actauly goes out of bounds on the last bubble, but jax pulls it back to last idx
    # since its the bubble we don't care that its randomly clipped to the last, but should probably change this
    #weights_repeated = jnp.reshape(weights, [num_stages, num_repeat, model_dim, model_dim])
    # TODO!!! lax.dynamic slice
    to_stack = [weights[layer_ids[stage],:,:] for stage in range(args.n_stages)]
    to_ret = jnp.concatenate(to_stack, axis=0)
    desired_shape = (args.n_stages,) + weights.shape[1:]
    to_ret = jnp.reshape(to_ret,desired_shape) # some singleton axes may have gotten flattened
    return to_ret

def run_one_iteration(state, shift, async_state, lir, loop_iteration, weights):
   stages_in = get_iteration_inputs(loop_iteration, args.n_microbatches, args.n_stages, state, async_state)
   stages_in = select_state_or_input(stages_in, shift)
   my_print(f"Stages in: {jnp.ravel(stages_in)}")
   weights_stage = get_weights_stage(weights, loop_iteration)
   output = jax.vmap(stage, in_axes=0, out_axes=0,
                        spmd_axis_name='stage')(weights_stage, stages_in)
   new_state, new_shift, new_async_state, new_lir = get_new_loop_state(output, state, async_state, lir, loop_iteration)
   return new_state, new_shift, new_async_state, new_lir

def permute_ms_dim(state):
    # How come I don't see this function in praxis?
    ms_size = state.shape[1]
    # More accurately land_idx = microbatches * (r - 1) + num_stages - 1 % ms, but ms | microbatches
    land_idx = (args.n_stages - 1) % ms_size # first_finish % ms_size (really first_finish - 1 is the idx we careabout)
    permutation = (np.arange(ms_size) + land_idx) % ms_size
    state = state[:,permutation]
    return state

def init_states(inputs):
    # Initialize shift and state
    shift = jnp.zeros((args.n_stages,) + inputs.shape[1:]) # equivalently inputs.shape[1:] is microshape
    state = jnp.reshape(inputs, (args.n_stages, args.n_microbatches // args.n_stages) + inputs.shape[1:])
    # [num_stages, num_micro, micro_size, ...]
    async_state = jnp.zeros((args.n_stages,) + inputs.shape ) # This is huge, is this correct?
    lir = shift
    return state, shift, async_state, lir
   

def run_pipeline(weights, inputs):
    state, shift, async_state, lir = init_states(inputs)

    #total_iterations = microbatches + num_repeat * num_stages  - 1
    total_iterations = args.n_microbatches * args.num_repeat + args.n_stages  - 1 # What? Shoulnd't this be num_stages * num_repeat + micro - 1
    #breakpoint()
    for loop_iteration in range(total_iterations):
       my_print(f"Starting loop {loop_iteration}")
       my_print(f"shift:{jnp.ravel(shift)}")
       #my_print(f"state: {jnp.ravel(state)}")
       if yes_print:
        ss = jnp.reshape(state, [4,2])
        my_print(f"ss: {ss}")
        ras = jnp.reshape(async_state, [4,8])
        my_print(f" as: {ras}")
        my_print(f"lir: {jnp.ravel(lir)}")
       state, shift, async_state, lir = run_one_iteration(state, shift, async_state, lir, loop_iteration, weights)

    my_print("Final output")
    my_print(f"shift:{jnp.ravel(shift)}")
    if yes_print:
        my_print(f"state: {jnp.reshape(jnp.ravel(state),[4,2])}")
    # reshape state to match input shape
    #state = jnp.transpose(state, axes=(0,2,1,3,4)) # holy crap
    #qqq = jnp.transpose(state, axes=(2,3,4,1,0))
    state_perm = permute_ms_dim(state)

    state = jnp.reshape(state_perm, (args.n_microbatches,) + state.shape[2:])
    return state # this can be reshaped to match input at some point


######################     Begin main      #################


def get_weights_random():
    # Assuming layer_i looks like output = inputs[micro_id,:,:,:] * weights[i,:,:] --> x_out = jnp.einsum('bse,eh->bsh',x,weights)
    weights_random_shape = jnp.array([args.n_stages * args.num_repeat, args.features, args.features]) # more realistic:  layers x embed x hidden, etc
    k = jax.random.PRNGKey(1)
    return jax.random.normal(k,weights_random_shape, dtype=jnp.float32)

def get_weights_debug_unique():
    weights = list()
    weights_debug_shape = [args.n_stages * num_repeat, args.microbatch_size, args.features, args.features]
    for i in range(jnp.prod(weights_debug_shape)):
       weights.append((i+1) * 10**(i+1))
    weights = jnp.array(weights, dtype=jnp.float32)
    weights = jnp.reshape(weights, weights_debug_shape)
    return weights

def get_inputs_random():
    micro_shape = [args.microbatch_size, args.sequence, args.features] # realistic
    test_input_shape = [args.n_microbatches] + micro_shape # [microbatches, microbatch_size, seq_len, model_dim]
    k = jax.random.PRNGKey(2)
    return jax.random.normal(k,test_input_shape, dtype=jnp.float32)

def get_weights_debug():
   # Assuming layer_i looks like output = inputs[micro_id,:,:,:] + weights[i,:,:,:] --> x_out = x_out = inputs + weights
    weights_debug_shape = [args.n_stages * args.num_repeat, args.microbatch_size, args.features, args.features]
    return 100 + jnp.zeros(weights_debug_shape, dtype=jnp.float32)
   
def get_inputs_debug():
    test_inputs_shape = jnp.array([args.n_microbatches] + micro_shape)
    test_inputs = jnp.reshape(jnp.arange(jnp.prod(test_inputs_shape), dtype=jnp.float32), test_inputs_shape)
   


# Sizes
num_stages = 4
microbatches = 8
microbatch_size = 5
seq_len = 2048
model_dim = 2560
total_batch = microbatches * microbatch_size
num_repeat = 3

yes_print = False
sum_layer = False

micro_shape = [microbatch_size, seq_len, model_dim] # realistic
#micro_shape = [microbatch_size] # great for debugging state transformations
#micro_shape = [microbatch_size, model_dim] # middle ground for debugging running with weights

k = jax.random.PRNGKey(1)

test_inputs = get_inputs_random()
weights = get_weights_random()

# Configure sharding
pipeline_axis = 4
dp_axis = 1
devices = mesh_utils.create_device_mesh((pipeline_axis, dp_axis))
mesh = Mesh(devices, axis_names=('stage', 'data'))

weight_sharding = S('stage', None, None) # weight sharded over stage
input_sharding = S('data', None, None, None)   # inputs sharded over batch
result_sharding = S('data', None, None, None)  # output sharded over batch

#weights = jax.device_put(weights, weight_sharding)
#jax.debug.visualize_array_sharding(weights)


####### Start testing ###########

# Test get_weights_stage
if 0:
    ws = get_weights_stage(weights, 0)


# Test run_one_iteration
# Initialize shift and state
if 0:
    shift = jnp.zeros((args.n_stages,) + test_inputs.shape[1:]) # equivalently inputs.shape[1:] is microshape
    state = jnp.reshape(test_inputs, (args.n_stages, args.n_microbatches // args.n_stages) + test_inputs.shape[1:])
    new_state, new_shift = run_one_iteration(state, shift, 0, weights)


# Test get_iteration input + select_state = stages_in
if 0:
    loop_iteration = 0
    state, shift, async_state, lir = init_states(1.0 + test_inputs)
    stages_in = get_iteration_inputs(loop_iteration, args.n_microbatches, args.n_stages, state, async_state)
    stages_in = select_state_or_input(stages_in, shift)


# Test get_new_loop_state
if 0:
    loop_iteration = 0
    state, shift, async_state, lir = init_states(1.0 + test_inputs)
    output = shift
    new_state, new_shift, new_async_state, new_lir = get_new_loop_state(output, state, async_state, lir, loop_iteration)
    assert new_state.shape == state.shape
    assert new_shift.shape == shift.shape
    assert new_async_state.shape == async_state.shape
    assert new_lir.shape == lir.shape

# Test get_weights_stage
if 0:
   weights = jnp.reshape(jnp.arange(8),weights_shape)
   for loop_iteration in range(19):
      weights_stage = get_weights_stage(weights, loop_iteration)
      print(f"iter {loop_iteration}: weights {jnp.ravel(weights_stage)}")

# Test run pipeline (no jit)
if 0:

    weights = get_weights_debug()
    inputs = get_inputs_debug()
    print(f"weights: {jnp.ravel(weights)}")
    print(f"inputs: {jnp.ravel(weights)}")

    outputs = run_pipeline(weights, test_inputs)
    #print(f"{outputs=}")

# Test jitted E2E
def rawr():
    if 1:
        weights = get_weights_random()
        test_inputs = get_inputs_random()

        output_jit = jax.jit(run_pipeline,
                    in_shardings=((weight_sharding, input_sharding)),
                    out_shardings=result_sharding)

        output_pipeline = output_jit(weights, test_inputs)
        # [Microbatch, microsize, seq embed] -> [Batch, Seq, Embed]
        output_pipeline= jnp.reshape(output_pipeline, (args.batch_size,) + output_pipeline.shape[2:])

        def reg_layer(weights, input):
            for layer_idx in range(weights.shape[0]):
                input = layer(weights[layer_idx,:,:], input)
            return input

        # Reshape batched_inputs from [micro,micro_size,...] to [batch,...]
        batched_inputs = jnp.reshape(test_inputs, (args.batch_size,) + test_inputs.shape[2:])
        regular_output = reg_layer(weights, batched_inputs)

        diff_norm = jnp.linalg.norm(output_pipeline - regular_output)
        print(f"{diff_norm=}")

        regular_norm = jnp.linalg.norm(regular_output)
        print(f"{regular_norm=}")

        output_pipeline_norm = jnp.linalg.norm(output_pipeline)
        print(f"{output_pipeline_norm=}")

        yes_print = False
        my_print(f"regular {jnp.ravel(regular_output)}")
        my_print(f"pipeline {jnp.ravel(regular_output)}")

# Test 1 stage of vmap
if 0:
    sum_layer = False
    weights = get_weights_random()
    inputs = get_inputs_random()
    print(f"{weights=}")
    print(f"{inputs=}")

    batched_inputs = jnp.reshape(inputs, (args.batch_size,) + test_inputs.shape[2:])
    layer_1_reg = layer(weights[0,:,:], batched_inputs)
    #print(f"{layer_1_reg=}")

    loop_iteration = 0
    state, shift, async_state, lir = init_states(inputs)
    stages_in = get_iteration_inputs(loop_iteration, args.n_microbatches, args.n_stages, state, async_state)
    stages_in = select_state_or_input(stages_in, shift)
    #my_print(f"Stages in: {jnp.ravel(stages_in)}")
    weights_stage = get_weights_stage(weights, loop_iteration)
    pipeline_output = jax.vmap(stage, in_axes=0, out_axes=0,
                        spmd_axis_name='stage')(weights_stage, stages_in)
    print(f"{pipeline_output=}")

    
    batched_state = batched_inputs
    for layer_idx in range(weights.shape[0]):
       batched_state = layer(weights[layer_idx,:,:], batched_state)
       print(f"layer {layer_idx}, batched_state: {jnp.ravel(batched_state)}")


    total_iterations = args.n_microbatches * args.num_repeat + args.n_stages
    for loop_iteration in range(total_iterations):
       my_print(f"Starting loop {loop_iteration}")
       my_print(f"shift:{jnp.ravel(shift)}")
       #my_print(f"state: {jnp.ravel(state)}")
       if yes_print:
        ss = jnp.reshape(state, [4,2])
        my_print(f"ss: {ss}")
        ras = jnp.reshape(async_state, [4,8])
        my_print(f" as: {ras}")
        my_print(f"lir: {jnp.ravel(lir)}")
       state, shift, async_state, lir = run_one_iteration(state, shift, async_state, lir, loop_iteration, weights)

    
       

if 1:
    import timing_util
    weights = get_weights_random()
    inputs = get_inputs_random()
    print(f"weights: {jnp.ravel(weights)}")
    print(f"inputs: {jnp.ravel(inputs)}")

    timing_util.simple_timeit(run_pipeline, weights, test_inputs, task = "run_pipeline")
    #outputs = run_pipeline(weights, test_inputs)
    #print(f"{outputs=}")


def main() -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    parser = argparse.ArgumentParser(description='Pipeline Parallelism Options')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_stages', type=int, default=4)
    parser.add_argument('--n_microbatches', type=int, default=4)
    parser.add_argument('--pipeline_axis', type=int, default=4)
    parser.add_argument('--dp_axis', type=int, default=1)
    parser.add_argument('--features', type=int, default=16)
    parser.add_argument('--sequence', type=int, default=2048)
    parser.add_argument('--num_repeat', type=int, default=1)

    global args
    args = parser.parse_args()
    args.microbatch_size = args.batch_size // args.n_microbatches


    # Necessary artifacts for the good stuff
    #pipeline_func = get_pipelint_jit()
    #weights, inputs, targets = get_weights_and_inputs()

    rawr()

    #assert_same_output_and_grad(reg_matmuls, pipeline_func, targets, weights, inputs)

    #timing_util.simple_timeit(pipeline_func, weights, inputs, tries = 3, task = 'basic_pp')



if __name__ == "__main__":
  main()