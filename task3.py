# Import the necessary modules and libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, DistributedSampler

# Define the hyperparameters and constants
num_epochs = 10 # number of training epochs
batch_size = 64 # number of training batch size
lr = 0.001 # learning rate
world_size = 4 # number of processes for DDP or FSDP
fsdp_config = {"mixed_precision": True, "cpu_offload": True} # configuration for FSDP

# Define the model class
class GPT2(nn.Module):
    # The same model class as defined in the previous task
    ...

# Define the training function
def train(rank, world_size, parallel_mode):
    # Initialize the process group for DDP or FSDP
    if parallel_mode in ["DDP", "FSDP"]:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Set the device to the current rank
    device = torch.device(rank)
    
    # Create the model and move it to the device
    model = GPT2()
    model.to(device)
    
    # Wrap the model with DDP or FSDP
    if parallel_mode == "DDP":
        model = DDP(model, device_ids=[rank])
    elif parallel_mode == "FSDP":
        model = FSDP(model, process_group=dist.group.WORLD, **fsdp_config)
    
    # Create the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Load the MNIST dataset
    train_dataset = MNIST(root="data", train=True, transform=ToTensor(), download=True)
    test_dataset = MNIST(root="data", train=False, transform=ToTensor())
    
    # Create the data loaders with distributed samplers for DDP or FSDP
    if parallel_mode in ["DDP", "FSDP"]:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # Train the model for the given number of epochs
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()
        # Iterate over the training batches
        for inputs, labels in train_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Compute the loss
            loss = loss_fn(outputs, labels)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # Print statistics
            print(f"Rank {rank}, Epoch {epoch}, Loss {loss.item()}")
        # Set the model to evaluation mode
        model.eval()
        # Evaluate the model on the test set
        correct = 0
        total = 0
        with torch.no_grad():
            # Iterate over the test batches
            for inputs, labels in test_loader:
                # Move the inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(inputs)
                # Get the predictions
                _, preds = torch.max(outputs, 1)
                # Update the accuracy
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        # Print the accuracy
        print(f"Rank {rank}, Epoch {epoch}, Accuracy {correct / total}")
    
    # Destroy the process group for DDP or FSDP
    if parallel_mode in ["DDP", "FSDP"]:
        dist.destroy_process_group()

# Define the main function
def main():
    # Choose the parallel mode: "SG" for single GPU, "DDP" for distributed data parallel, or "FSDP" for fully sharded data parallel
    parallel_mode = "SG"
    
    # Run the training function on a single GPU
    if parallel_mode == "SG":
        train(0, 1, parallel_mode)
    
    # Spawn multiple processes and run the training function on multiple GPUs using DDP or FSDP
    else:
        mp.spawn(train, args=(world_size, parallel_mode), nprocs=world_size, join=True)

# Run the main function
if __name__ == "__main__":
    # Set the environment variables for DDP or FSDP
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
