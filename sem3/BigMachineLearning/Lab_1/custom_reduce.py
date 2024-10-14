import os
import time
import torch
import torch.distributed as dist

WORLD_SIZE = int(os.environ['WORLD_SIZE'])
RANK = int(os.environ['SLURM_PROCID'])
dist.init_process_group("gloo", rank=RANK, world_size=WORLD_SIZE)

def custom_reduce(tensor, destination, operation):
    if RANK == destination:
        result = tensor.clone()
        for i in range(WORLD_SIZE):
            if i != destination:
                temp = torch.zeros_like(tensor)
                dist.recv(temp, i)
                result = operation(result, temp)
        return result
    else:
        dist.send(tensor, destination)
    return None

def custom_all_reduce(tensor, operation):
    result = custom_reduce(tensor, 0, operation)
    if RANK != 0:
        result = torch.zeros_like(tensor)
        dist.recv(result, 0)
    else:
        for i in range(1, WORLD_SIZE):
            dist.send(result, i)
    return result

tensor = torch.tensor([RANK], dtype=torch.float32)

start_time = time.time()
reduced_tensor = custom_reduce(tensor, 0, torch.add)
end_time = time.time()
print(f"Rank {RANK}: Custom reduce time = {end_time - start_time} seconds", flush=True)
if RANK == 0:
    print(f"Rank 0: Custom reduce result = {reduced_tensor.item()}", flush=True)

tensor = torch.tensor([RANK], dtype=torch.float32)

start_time = time.time()
dist.reduce(tensor, 0, op=dist.ReduceOp.SUM)
end_time = time.time()
print(f"Rank {RANK}: Torch reduce time = {end_time - start_time} seconds", flush=True)
if RANK == 0:
    print(f"Rank 0: Torch reduce result = {tensor.item()}", flush=True)

tensor = torch.tensor([RANK], dtype=torch.float32)

start_time = time.time()
all_reduced_tensor = custom_all_reduce(tensor, torch.add)
end_time = time.time()
print(f"Rank {RANK}: Custom all_reduce time = {end_time - start_time} seconds", flush=True)
print(f"Rank {RANK}: Custom all_reduce result = {all_reduced_tensor.item()}", flush=True)

tensor = torch.tensor([RANK], dtype=torch.float32)

start_time = time.time()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
end_time = time.time()
print(f"Rank {RANK}: Torch all_reduce time = {end_time - start_time} seconds", flush=True)
print(f"Rank {RANK}: Torch all_reduce result = {tensor.item()}", flush=True)
