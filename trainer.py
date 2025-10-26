import argparse

from torch.utils.data import DataLoader
import wandb
from utils import make_vgg, get_dataset, get_device
from utils import train_model

import warnings
warnings.filterwarnings("ignore", category=Warning, module="pydantic")

# Main training function used by WandB sweep agent
def train(config=None):
    """Train a VGG model with the given configuration (used by WandB sweep)."""
    # Set device (GPU if available, otherwise CPU)
    device = get_device()
    print(f"Using device: {device}")

    # Load CIFAR-10 dataset with appropriate transforms
    trainset, testset = get_dataset()
    
    # Initialize WandB run with the provided configuration
    with wandb.init(config=config) as wanrun:

        config = wanrun.config  # Get the sweep configuration parameters
        # Create data loaders for training and validation
        trainloader = DataLoader(
            trainset, batch_size=config.batch_size, 
            shuffle=True, num_workers=2  # Shuffle training data for better learning
        )
        testloader = DataLoader(
            testset, batch_size=config.batch_size, 
            shuffle=False, num_workers=2  # No shuffling needed for validation
        )
        # Create VGG model based on sweep configuration
        model = make_vgg(config)
        model = model.to(device)  # Move model to appropriate device (GPU/CPU)
        # Train the model and log metrics to WandB
        train_model(model, trainloader, testloader, config, device, wanrun)

# Entry point for command-line execution of hyperparameter sweeps
def main():
    """Main function to set up and run WandB hyperparameter sweep."""
    # Parse command line arguments for sweep configuration
    parser = argparse.ArgumentParser(description='Train VGG models with different configurations')
    parser.add_argument('--sweep_id', type=str, help='WandB sweep ID')
    parser.add_argument('--project', type=str, help='WandB project name')
    parser.add_argument('--count', type=int, help='Number of runs for the sweep', default=5)
    args = parser.parse_args()

    print(f"Starting WandB sweep with ID: {args.sweep_id} in project: {args.project}")

    # Start WandB agent to run hyperparameter sweep
    # The agent will run 'train' function multiple times with different configs
    wandb.agent(args.sweep_id, train, project=args.project, count=args.count)


if __name__ == '__main__':
    main()