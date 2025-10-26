import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import get_dataset, get_device
from utils import test_model


# Main function for testing saved models on CIFAR-10 test set
def main():
    """Main function to test a saved VGG model on CIFAR-10 test set."""
    # Parse command line arguments for model testing
    parser = argparse.ArgumentParser(description='Test saved VGG model')
    parser.add_argument('model_path', type=str, help='Path to the saved model (.pth file)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing')
    
    args = parser.parse_args()
    
    # Set device (GPU if available, otherwise CPU)
    device = get_device()
    print(f"Using device: {device}")
    
    # Load the pre-trained model from the specified path
    print(f"Loading model from: {args.model_path}")
    model = torch.load(args.model_path, map_location=device,weights_only=False)
    model = model.to(device)  # Move model to appropriate device
    model.eval()  # Set model to evaluation mode (disables dropout, batchnorm updates)
    
    # Load CIFAR-10 test dataset with normalization transforms
    _, testset = get_dataset(include_trainset=False)  # Only need test set for inference
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Run inference on the test set
    criterion = nn.CrossEntropyLoss()  # Loss function for classification
    test_loss, test_accuracy = test_model(model, testloader, criterion, device)
    
    # Print final test results
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.2f}%")


if __name__ == '__main__':
    main()