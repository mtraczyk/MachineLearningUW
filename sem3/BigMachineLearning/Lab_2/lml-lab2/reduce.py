import os
import time
import torch
import torch.distributed as dist

WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
dist.init_process_group("gloo", rank=RANK, world_size=WORLD_SIZE)

message = torch.tensor(RANK)
print(f"Rank {RANK}: Sending {message.item()}")
dist.reduce(message, 0, dist.ReduceOp.SUM)

if RANK == 0:
  print(f"Rank 0: Sum of all messages is {message.item()}")