"""
Hyperparameter tuning script for BaselineMLP using wandb sweeps.

This script uses wandb's sweep functionality to test different hyperparameters
for the BaselineMLP model with standard MSE loss (no custom loss function).

Hyperparameters tested:
- Hidden layer sizes
- Learning rate
- Number of epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_models import BaselineMLP


def load_data(train_path, val_path, batch_size=64):
    """Load training and validation datasets."""
    train_data = np.load(train_path, allow_pickle=True)
    val_data = np.load(val_path, allow_pickle=True)

    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    metadata = train_data['metadata'].item()

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, metadata


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_x)
            loss = criterion(output, batch_y)

            total_loss += loss.item()

    return total_loss / len(val_loader)


def train():
    """Main training function for wandb sweep."""
    # Initialize wandb run
    run = wandb.init()

    # Get hyperparameters from wandb config
    config = wandb.config

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    train_path = os.path.join(data_dir, 'train_dataset.npz')
    val_path = os.path.join(data_dir, 'val_dataset.npz')

    # Load data
    train_loader, val_loader, metadata = load_data(
        train_path, val_path, batch_size=config.batch_size
    )

    # Parse hidden_dims from config
    hidden_dims = config.hidden_dims

    # Create model
    model = BaselineMLP(
        input_dim=metadata['input_dim'],
        output_dim=metadata['output_dim'],
        hidden_dims=hidden_dims
    ).to(device)

    # Setup optimizer and loss (using standard MSE loss, no custom loss function)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    wandb.log({'total_parameters': total_params})

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Log metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
        })

        # Track best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # Log final best validation loss
    wandb.log({'best_val_loss': best_val_loss})

    run.finish()


# Define sweep configuration
sweep_config = {
    'method': 'grid',  # Use grid search to explore all combinations
    'metric': {
        'name': 'best_val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'hidden_dims': {
            'values': [
                [32, 32],           # Small network
                [64, 64],           # Medium network
                [64, 128, 64],      # Default from BaselineMLP
                [128, 128],         # Larger network
                [128, 256, 128],    # Large network
            ]
        },
        'learning_rate': {
            'values': [1e-2, 1e-3, 1e-4]
        },
        'epochs': {
            'values': [50, 100, 150]
        },
        'batch_size': {
            'value': 64  # Fixed batch size
        }
    }
}


def main():
    """Main function to run wandb sweep."""
    print("=" * 60)
    print("BASELINE MLP HYPERPARAMETER SWEEP")
    print("Using wandb sweep with standard MSE loss")
    print("=" * 60)

    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project="ese562-baseline-mlp-sweep"
    )

    print(f"\nSweep ID: {sweep_id}")
    print("Starting sweep agent...")

    # Run sweep agent
    # This will run train() for each combination of hyperparameters
    wandb.agent(sweep_id, function=train)

    print("\n" + "=" * 60)
    print("SWEEP COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
