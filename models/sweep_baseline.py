"""
Hyperparameter tuning script for BaselineMLP using wandb sweeps.

This script uses wandb's sweep functionality to test different hyperparameters
for the BaselineMLP model with standard MSE loss (no custom loss function).

Hyperparameters tested:
- Hidden layer sizes
- Learning rate
- Number of epochs

Usage:
    python sweep_baseline.py [--project PROJECT_NAME] [--data-dir DATA_DIR]

Environment variables:
    WANDB_PROJECT: wandb project name (default: ese562-baseline-mlp-sweep)
    DATA_DIR: path to data directory (default: ../data)
"""

import argparse
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

# Global variable to store data directory path
_data_dir = None


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
    global _data_dir

    # Initialize wandb run
    run = wandb.init()

    # Get hyperparameters from wandb config
    config = wandb.config

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data paths - use global _data_dir if set, otherwise use default
    if _data_dir is not None:
        data_dir = _data_dir
    else:
        data_dir = os.environ.get(
            'DATA_DIR',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        )
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


def get_total_combinations():
    """Calculate total number of hyperparameter combinations."""
    hidden_dims_count = len(sweep_config['parameters']['hidden_dims']['values'])
    lr_count = len(sweep_config['parameters']['learning_rate']['values'])
    epochs_count = len(sweep_config['parameters']['epochs']['values'])
    return hidden_dims_count * lr_count * epochs_count


def main():
    """Main function to run wandb sweep."""
    global _data_dir

    parser = argparse.ArgumentParser(
        description='Run wandb sweep for BaselineMLP hyperparameter tuning'
    )
    parser.add_argument(
        '--project',
        type=str,
        default=os.environ.get('WANDB_PROJECT', 'ese562-baseline-mlp-sweep'),
        help='wandb project name (default: ese562-baseline-mlp-sweep)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=os.environ.get(
            'DATA_DIR',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        ),
        help='path to data directory containing train_dataset.npz and val_dataset.npz'
    )
    args = parser.parse_args()

    # Set global data directory
    _data_dir = args.data_dir

    # Calculate total combinations for grid search
    total_combinations = get_total_combinations()

    print("=" * 60)
    print("BASELINE MLP HYPERPARAMETER SWEEP")
    print("Using wandb sweep with standard MSE loss")
    print("=" * 60)
    print(f"Project: {args.project}")
    print(f"Data directory: {args.data_dir}")
    print(f"Total hyperparameter combinations: {total_combinations}")
    print("=" * 60)

    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project=args.project
    )

    print(f"\nSweep ID: {sweep_id}")
    print("Starting sweep agent...")

    # Run sweep agent with count limit for grid search
    # This ensures all combinations are run exactly once
    wandb.agent(sweep_id, function=train, count=total_combinations)

    print("\n" + "=" * 60)
    print("SWEEP COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
