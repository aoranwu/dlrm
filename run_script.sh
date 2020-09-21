export MASTER_PORT=29500
export MASTER_ADDR=10.10.1.2
export WORLD_SIZE=2
export RANK=0
# python /usr/local/lib/python3.6/dist-packages/torch/distributed/launch.py --nnode=2 --node_rank=0 --nproc_per_node=1 dlrm_s_pytorch.py --mini-batch-size=2 --data-size=6
python dlrm_s_pytorch.py --mini-batch-size=10 --data-size=40 --use-gpu --enable-profiling --out-dir=prof --plot-compute-graph --print-time --dist-backend=gloo --arch-embedding-size=10-6-2-8-10-4-10-4

