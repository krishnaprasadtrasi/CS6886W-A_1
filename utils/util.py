import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .model import VGG
from .transformers import Cutout, CIFAR10Policy


# factory function to get activation functions by name
def get_activation_function(activation_name):
    """Get activation function by name - supports various activation types."""
    if activation_name.lower() == 'relu':
        return nn.ReLU(inplace=True)  # Most common, computationally efficient
    elif activation_name.lower() == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01, inplace=True) 
    elif activation_name.lower() == 'elu':
        return nn.ELU(inplace=True)  
    elif activation_name.lower() == 'gelu':
        return nn.GELU()  
    elif activation_name.lower() == 'swish' or activation_name.lower() == 'silu':
        return nn.SiLU(inplace=True)  
    elif activation_name.lower() == 'mish':
        return nn.Mish(inplace=True)  
    elif activation_name.lower() == 'tanh':
        return nn.Tanh()  # Output range [-1, 1]
    elif activation_name.lower()  == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"Activation function {activation_name} not supported")


# Constructs sequential layers for VGG architecture from configuration
def make_layers(cfg, batch_norm=False, activation='relu'):
    """Create VGG layers based on configuration string."""
    layers = []
    in_channels = 3  
    
    # Parse configuration to build layers
    for v in cfg:
        if v == 'M':
            # Max pooling layer (reduces spatial dimensions)
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            # Average pooling layer (alternative to max pooling)
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        elif v == 'D':
            # Dropout layer (prevents overfitting)
            layers += [nn.Dropout(0.5)]
        elif v == 'F':
            # Flatten layer (converts 2D to 1D for fully connected layers)
            layers += [nn.Flatten()]
        else:
            # Convolutional layer with specified number of output channels
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # Add batch normalization for training stability
                layers += [conv2d, nn.BatchNorm2d(v), get_activation_function(activation)]
            else:
                layers += [conv2d, get_activation_function(activation)]
            in_channels = v  # Update input channels for next layer
    return nn.Sequential(*layers)

# Factory function to create VGG model instances with custom configurations
def make_vgg(config):
    """Create a VGG model instance based on the provided configuration."""
    # Build the feature extraction layers (convolutional part)
    layers = make_layers(config.vgg_architecture, batch_norm=config.batch_norm, activation=config.activation)
    # Create complete VGG model with classifier layers
    return VGG(layers, num_classes=config.num_classes)

# Loads and preprocesses CIFAR-10 dataset with appropriate transforms
def get_dataset(include_trainset=True):
    """Load and prepare the CIFAR-10 dataset with appropriate transforms."""

    # Test set transforms: only normalization (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor [0,1]
        # Normalize with CIFAR-10 mean and std for each channel
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # Load CIFAR-10 test set
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    trainset = None
    if include_trainset:
        # Training set transforms: augmentation + normalization
        transform_train = transforms.Compose([
                # Data augmentation techniques to improve generalization
                transforms.RandomCrop(32, padding=4),  # Random crop with padding
                transforms.RandomHorizontalFlip(),     
                CIFAR10Policy(),
                transforms.ToTensor(),  # Convert to tensor
                # Same normalization as test set
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                Cutout(n_holes=1, length=16)
            ])
        # Load CIFAR-10 training set                
        trainset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
    return trainset, testset


# Factory function for creating optimizers based on configuration settings
def get_optimizer(model, config):
    """Create and return the specified optimizer with model parameters."""
    # Extract optimizer configuration parameters
    optimizer_name = config.optimizer
    learning_rate = config.learning_rate
    momentum = config.momentum
    weight_decay = config.weight_decay

    # Return the appropriate optimizer based on configuration
    if optimizer_name.lower() == 'adam':
        # Adam
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        # SGD
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'sgd_nesterov':
        return optim.SGD(model.parameters(), lr=learning_rate,
                         momentum=momentum, nesterov=True, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        # AdamW
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'rmsprop':
        # RMSprop
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'adagrad':
        # Adagrad
        return optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'adadelta':
        # Adadelta
        return optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'adamax':
        # Adamax
        return optim.Adamax(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'nadam':
        return optim.NAdam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")
    

# Detects and returns the best available computing device (GPU or CPU)
def get_device():
    """Determine and return the best available device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')