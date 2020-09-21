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

my_size is world_size

get_my_slice(n): given a total number n (batch size for data parallel, # of embedding tables for model parallel), determine which slice to deal with based on my_rank

get_split_length(n): return len of own slice and the whole splits (my_len, splits)


RANK is the node rank(?) My rank is the global rank. get from dist.get_rank

Why print only happens at the master machine

Use of alltoall in distributed_forward()
lS_o[i], lS_i[i] contains the whole batch of data for the ith embedding table

All2All_Scatter_Req
All2All_Scatter_Wait

--arch-embedding-size: # of entries for each embedding tables  
e.g. 4-4-3-5, four embedding tables in total, each have 4,4,3,5 entries in total

--arch-sparse-feature-size: embedding table feature dim

N: batch_size
E: # of feature dims
a2ai.lS= len(inputs) # of embedding tables
a2ai.gSS: how these embedding tables are split across different ranks. It's a list of embedding table # on each rank
a2ai.lN: local batch
a2ai.gNS: global batch split, also a list
inputs(ly): (embedding_table_num on this rank, batch_size, feature_dim)
a2ai.S: total embedding table num
a2ai.lS: local embedding table num

lS_i: (embedding_table, batch)
each lS_i[k] is the full batch for embedding table k
use lS_o to split the batch
eg: lS_i = {list: 3} [tensor([1, 0, 1]), tensor([0, 1]), tensor([1, 0])]
lS_o = {Tensor: 3} tensor([[0, 1],\n        [0, 1],\n        [0, 1]])
each offset in lS_o[k] specify a sample in embedding k
lS_o[k] along with lS_i[k] specify the whole batch for embedding table k

for ly: ly[k] is the result of whole batch for embedding table k 
ly[k][0] is the result of the first sample of the batch for embedding table k

torch.cat: remove one dim(list/tuple dim), then add along the given dim
2*2 2*1 -> dim=1, 2*3 add along the given dim

logging added at dist:218, 236, 252, 270

DDP allreduce:
torch.nn.parallel.distributed.py:527
self.reducer.prepare_for_backward()


process_group_->allreduce()
/home/aoran/pytorch/torch/csrc/distributed/c10d/reducer.cpp:678

/home/aoran/pytorch/torch/lib/c10d/ProcessGroupGloo.cpp

/home/aoran/pytorch/third_party/gloo/gloo/allreduce.cc