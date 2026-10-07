#!/bin/bash
NUM_PROC=$1
shift 1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nproc_per_node=$NUM_PROC main.py "$@"