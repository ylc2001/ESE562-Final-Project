import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_models import get_model


class PowerFlowTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: PyTorch model to train
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            output = self.model(batch_x)
            loss = criterion(output, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                output = self.model(batch_x)
                loss = criterion(output, batch_y)

                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def train(self, train_loader, val_loader, num_epochs=100, learning_rate=1e-3,
              patience=15, save_dir='checkpoints'):
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        # Training loop
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 60)

        epochs_no_improve = 0

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                epochs_no_improve = 0

                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"  --> Saved best model (val_loss: {val_loss:.6f})")
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")

        # Save training history
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
            }, f)

        return self.train_losses, self.val_losses

    def plot_training_history(self, save_path=None):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('Training History', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")

        plt.show()


def load_data(train_path, val_path, batch_size=64):
    # Load datasets
    train_data = np.load(train_path, allow_pickle=True)
    val_data = np.load(val_path, allow_pickle=True)

    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    metadata = train_data['metadata'].item()

    print("Dataset loaded:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    print(f"  Input dimension: {X_train.shape[1]}")
    print(f"  Output dimension: {y_train.shape[1]}")
    print()

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


def main():
    """Main training function"""
    # Configuration
    config = {
        'model_name': 'baseline',  # 'baseline', 'improved', or 'physics'
        'batch_size': 64,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'patience': 15,
        'data_dir': '../data',
        'save_dir': 'checkpoints'
    }

    print("=" * 60)
    print("AC POWER FLOW NEURAL NETWORK TRAINING")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    print()

    # Prepare data paths
    train_path = os.path.join(config['data_dir'], 'train_dataset.npz')
    val_path = os.path.join(config['data_dir'], 'val_dataset.npz')

    # Load data
    train_loader, val_loader, metadata = load_data(
        train_path, val_path, config['batch_size']
    )

    # Create model
    model = get_model(
        model_name=config['model_name'],
        input_dim=metadata['input_dim'],
        output_dim=metadata['output_dim']
    )

    print(f"Model: {config['model_name']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create trainer
    trainer = PowerFlowTrainer(model)

    # Train model
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        save_dir=config['save_dir']
    )

    # Plot training history
    plot_path = os.path.join(config['save_dir'], 'training_history.png')
    trainer.plot_training_history(save_path=plot_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()