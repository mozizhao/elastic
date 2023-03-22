import os
import statistics
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

# curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py && sudo python3 install_gpu_driver.py && apt install python3-pip -y && pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu115

"""
(MNMC) Multiple Nodes Multi-GPU Cards Training
    with DistributedDataParallel and torch.distributed.launch
Try to compare with [snsc.py, snmc_dp.py & mnmc_ddp_mp.py] and find out the differences.
"""

"""
CUDA_VISIBLE_DEVICES=2,3,5,6,7 python3 -m torch.distributed.launch --nproc_per_node 5 ddp.py densenet121

NCCL_P2P_DISABLE=1 python3 -m torch.distributed.launch --nproc_per_node 4 ddp.py vgg19 128

NCCL_P2P_LEVEL=PHB python3 -m torch.distributed.launch --nproc_per_node 4 ddp.py vgg19 128

python -c "import torch;print(torch.cuda.nccl.version())
"""

BATCH_SIZE = 512
EPOCHS = 1

if __name__ == "__main__":
    batch_times = []

    # 0. set up distributed device
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    arch = sys.argv[2]
    if len(sys.argv) > 3:
        BATCH_SIZE = int(sys.argv[3])

    print(f"[init] == local rank: {local_rank}, model: {arch}, global rank: {rank} ==")

    # 1. define network
    model = torchvision.models.__dict__[arch]()
    model = model.to(device)
    # DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    trainset = torchvision.datasets.CIFAR100(root="./", train=True, download=True, transform=transforms.ToTensor())
    # DistributedSampler
    # we test single Machine with 2 GPUs
    # so the [batch size] for each process is 256 / 2 = 128
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        # num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
    )

    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # if rank == 0:
    #     print("            =======  Training  ======= \n")

    # 4. start to train
    model.train()

    for ep in range(1, EPOCHS + 1):
        train_loss = correct = total = 0
        # set sampler
        train_loader.sampler.set_epoch(ep)
        now = time.time()
        for idx, (inputs, targets) in enumerate(train_loader):
            batch_start = time.time()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

            torch.cuda.synchronize()

            if rank == 0:
                print(
                    "   == step: [{:3}/{}] [{}/{}] | loss: {:.3f} | acc: {:6.3f}% Cost: {}".format(
                        idx + 1,
                        len(train_loader),
                        ep,
                        EPOCHS,
                        train_loss / (idx + 1),
                        100.0 * correct / total,
                        time.time() - batch_start
                    )
                )
                batch_times.append(time.time() - batch_start)

        print("Epoch:", ep, " Cost:", time.time() - now)
    if rank == 0:
        print("\n            =======  Training Finished  ======= \n")

        batch_times.pop(0)
        batch_times.pop(-1)
        print(f'mean iteration time:{statistics.mean(batch_times)}')
