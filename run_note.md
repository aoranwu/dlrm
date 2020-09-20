# args.dist_backend should be set to 'nccl' or 'gloo'
# rank and size are all left to be -1 in dlrm: 629
# get rank and world_size from env RANK, WORLD SIZE in ext_dist: 71
# size in ext_dist.init_distributed() param means WORLD_SIZE
env var should set MASTER_PORT and MASTER_ADDR, or default addr is localhost, port is 29500
local_size only appears in line 82 and 81
my_local_rank and my_local_size set from env var, ext_dist: 88,89
init_process_group in line 95, with rank=rank, world_size=size
env var RANK WORLD_SIZE will be set by launch.py
Use dist.all_to_all_single (if exists) to do all_to_all. Check if alltoall_supported is True.
Can specify a2a_impl with env var DLRM_ALLTOALL_IMPL. Let it be "alltoall", then use dist.all_to_all_single or local scatter

dlrm:854: data parallel for bot_l and top_l. Allreduce is automatically handled by DDP
args.out_dir: output directory default will be .
args.enable_profiling: produce .prof file default is false --enable-profiling
ext_dist.get_split_length() and ext_dist.get_my_slice is used for embedding table dlrm:252,253,372   
get slice along batch (data parallel) or model dim (different tables)
Use distributed_forward dlrm:364
a2a is performed in dlrm:393
all_gather in dlrm:424

run from line 537
args:
--num-batches how many batches in total for each epoch can be specified to skip some iters
--nepochs epoch num
--use-gpu
--dist-backend "nccl" if left to be ""
--print-time 
--debug-mode dlrm:763 print model info
--enable-profiling
--plot-compute-graph
--out-dir default .
--data-generation default random

minibatch should be able to be evenly split across all gpus per process

dlrm_wrap used to put model to proper devices dlrm:894

loss_fn_wrap: put loss to proper divices dlrm:910

main training loop: dlrm:1013

Currently use All2All_Scatter_Req and All2All_Scatter_Wait to implement alltoall