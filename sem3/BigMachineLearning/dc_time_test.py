import os
import time
import torch
import torch.distributed as dist

start_time = time.time()

WORLD_SIZE = int(os.environ['WORLD_SIZE'])
RANK = int(os.environ['SLURM_PROCID'])
dist.init_process_group("gloo", rank=RANK, world_size=WORLD_SIZE)

if RANK == 0:
    dist.send(torch.tensor(42.0), 1)
elif RANK == 1:
    message = torch.zeros(1)
    dist.recv(message, 0)

end_time = time.time()

print(f"Rank {RANK}: Time taken = {end_time - start_time} seconds")
