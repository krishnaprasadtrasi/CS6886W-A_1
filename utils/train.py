from torch import nn
import torch
import wandb

from utils.util import get_optimizer
from utils.test import test_model

# Trains the model for one complete epoch through the training dataset
def train_epoch(model, trainloader, criterion, optimizer, device):
    """Train model for one epoch and return average loss and accuracy."""
    model.train()  # Set model to training mode (enables dropout, batchnorm updates)
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Iterate through all batches in the training set
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # Move data to the appropriate device (GPU/CPU)
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Reset gradients from previous iteration
        optimizer.zero_grad()
        # Forward pass: compute model predictions
        outputs = model(inputs)
        # Calculate loss between predictions and true labels
        loss = criterion(outputs, targets)
        # Backward pass: compute gradients
        loss.backward()
        # Update model parameters using computed gradients
        optimizer.step()
        
        # Accumulate loss for this batch
        running_loss += loss.item()
        # Get predicted class (highest probability)
        _, predicted = outputs.max(1)
        total += targets.size(0)  # Count total samples
        correct += predicted.eq(targets).sum().item()  # Count correct predictions
    
    # Return average loss and accuracy for this epoch
    return running_loss / len(trainloader), 100. * correct / total

# Complete training pipeline with validation and WandB logging
def train_model(model, trainloader, testloader, config, device, wanrun):
    """Train a model for the specified number of epochs and log metrics to WandB."""
    # Initialize loss function for classification
    criterion = nn.CrossEntropyLoss()
    # Get optimizer based on configuration (Adam, SGD, etc.)
    optimizer = get_optimizer(model, config)
    
    print(f"Training run name: {wanrun.name} ...")

    # Train for the specified number of epochs
    for epoch in range(config.epochs):
        # Train the model for one epoch
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        
        # Evaluate model on validation set
        val_loss, val_acc = test_model(model, testloader, criterion, device)

        # Log training and validation metrics to WandB
        wanrun.log({"train_loss": train_loss, "train_acc": train_acc,
                     "val_loss": val_loss, "val_acc": val_acc})
        
        print(f"Epoch [{epoch+1}/{config.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save the trained model as a WandB artifact
    model_artifact = wandb.Artifact("vgg", type="model", metadata=dict(config))

    # Save model to local file and add to WandB artifact
    torch.save(model, "model.pth")
    model_artifact.add_file("model.pth")
    # wanrun.save("model.pth")  # Save to WandB run files
    #wanrun.save("model.pth", policy="now")
    torch.save(model.state_dict(), "model.pth")

    wanrun.log_artifact(model_artifact)  # Log as versioned artifact