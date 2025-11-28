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


class PhysicsInformedLoss(nn.Module):
    """
    Custom loss function that incorporates physical constraints of power systems.

    This loss combines:
    1. MSE loss for prediction accuracy
    2. Voltage magnitude constraint penalty (0.9 <= Vm <= 1.1 p.u.)
    3. Reference bus angle constraint (slack bus Va = 0)
    4. Power balance residual penalty

    The power flow equations are:
    P_i = sum_j |V_i||V_j|(G_ij*cos(theta_ij) + B_ij*sin(theta_ij))
    Q_i = sum_j |V_i||V_j|(G_ij*sin(theta_ij) - B_ij*cos(theta_ij))

    where theta_ij = Va_i - Va_j
    """

    def __init__(self, ybus_real, ybus_imag, nbus=9, slack_bus=0,
                 vm_min=0.9, vm_max=1.1,
                 lambda_mse=1.0, lambda_vm=0.1, lambda_angle=0.1,
                 lambda_power=0.01):
        """
        Args:
            ybus_real: Real part of Ybus matrix (numpy array)
            ybus_imag: Imaginary part of Ybus matrix (numpy array)
            nbus: Number of buses in the system
            slack_bus: Index of slack/reference bus (0-indexed)
            vm_min: Minimum voltage magnitude (p.u.)
            vm_max: Maximum voltage magnitude (p.u.)
            lambda_mse: Weight for MSE loss
            lambda_vm: Weight for voltage magnitude constraint
            lambda_angle: Weight for reference angle constraint
            lambda_power: Weight for power balance constraint
        """
        super(PhysicsInformedLoss, self).__init__()

        # Store Ybus as buffers (not trainable parameters)
        self.register_buffer('G', torch.FloatTensor(ybus_real))  # Conductance
        self.register_buffer('B', torch.FloatTensor(ybus_imag))  # Susceptance

        self.nbus = nbus
        self.slack_bus = slack_bus
        self.vm_min = vm_min
        self.vm_max = vm_max

        # Loss weights
        self.lambda_mse = lambda_mse
        self.lambda_vm = lambda_vm
        self.lambda_angle = lambda_angle
        self.lambda_power = lambda_power

        # Base MSE loss
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets, inputs=None):
        """
        Compute physics-informed loss.

        Args:
            predictions: Predicted outputs [batch, 18] - [Va(9), Vm(9)]
            targets: True outputs [batch, 18] - [Va(9), Vm(9)]
            inputs: Input features [batch, 6] - [P_load(3), Q_load(3)]
                    (optional, for power balance calculation)

        Returns:
            Total loss combining all terms
        """
        batch_size = predictions.shape[0]

        # Extract voltage angles and magnitudes from predictions
        Va_pred = predictions[:, :self.nbus]  # [batch, 9]
        Vm_pred = predictions[:, self.nbus:]  # [batch, 9]

        # Extract targets
        Va_true = targets[:, :self.nbus]
        Vm_true = targets[:, self.nbus:]

        # 1. MSE Loss (primary loss)
        mse = self.mse_loss(predictions, targets)

        # 2. Voltage magnitude constraint penalty
        # Penalize Vm outside [vm_min, vm_max]
        vm_lower_violation = torch.relu(self.vm_min - Vm_pred)
        vm_upper_violation = torch.relu(Vm_pred - self.vm_max)
        vm_penalty = torch.mean(vm_lower_violation ** 2 + vm_upper_violation ** 2)

        # 3. Reference bus angle constraint
        # Slack bus angle should be close to 0
        angle_penalty = torch.mean(Va_pred[:, self.slack_bus] ** 2)

        # 4. Power balance residual (simplified physics constraint)
        # Calculate power injections from predicted voltages using power flow equations
        power_penalty = torch.tensor(0.0, device=predictions.device)

        if self.lambda_power > 0:
            # P_i = sum_j Vm_i * Vm_j * (G_ij * cos(Va_i - Va_j) + B_ij * sin(Va_i - Va_j))
            # Q_i = sum_j Vm_i * Vm_j * (G_ij * sin(Va_i - Va_j) - B_ij * cos(Va_i - Va_j))

            # Compute angle differences: theta_ij = Va_i - Va_j
            # Shape: [batch, nbus, nbus]
            Va_i = Va_pred.unsqueeze(2)  # [batch, 9, 1]
            Va_j = Va_pred.unsqueeze(1)  # [batch, 1, 9]
            theta_ij = Va_i - Va_j  # [batch, 9, 9]

            # Voltage products: Vm_i * Vm_j
            Vm_i = Vm_pred.unsqueeze(2)  # [batch, 9, 1]
            Vm_j = Vm_pred.unsqueeze(1)  # [batch, 1, 9]
            Vm_prod = Vm_i * Vm_j  # [batch, 9, 9]

            # Power flow calculations
            cos_theta = torch.cos(theta_ij)
            sin_theta = torch.sin(theta_ij)

            # P_i = sum_j Vm_i * Vm_j * (G_ij * cos + B_ij * sin)
            P_calc = torch.sum(Vm_prod * (self.G * cos_theta + self.B * sin_theta), dim=2)

            # Q_i = sum_j Vm_i * Vm_j * (G_ij * sin - B_ij * cos)
            Q_calc = torch.sum(Vm_prod * (self.G * sin_theta - self.B * cos_theta), dim=2)

            # Similarly for targets
            Va_true_i = Va_true.unsqueeze(2)
            Va_true_j = Va_true.unsqueeze(1)
            theta_ij_true = Va_true_i - Va_true_j

            Vm_true_i = Vm_true.unsqueeze(2)
            Vm_true_j = Vm_true.unsqueeze(1)
            Vm_prod_true = Vm_true_i * Vm_true_j

            cos_theta_true = torch.cos(theta_ij_true)
            sin_theta_true = torch.sin(theta_ij_true)

            P_true = torch.sum(Vm_prod_true * (self.G * cos_theta_true + self.B * sin_theta_true), dim=2)
            Q_true = torch.sum(Vm_prod_true * (self.G * sin_theta_true - self.B * cos_theta_true), dim=2)

            # Power balance penalty: calculated powers should match
            power_penalty = torch.mean((P_calc - P_true) ** 2 + (Q_calc - Q_true) ** 2)

        # Total loss
        total_loss = (self.lambda_mse * mse +
                      self.lambda_vm * vm_penalty +
                      self.lambda_angle * angle_penalty +
                      self.lambda_power * power_penalty)

        return total_loss


def build_ybus_matrix():
    """
    Build the Ybus admittance matrix for the 9-bus system.
    Returns real and imaginary parts separately.
    """
    nbus = 9

    # Branch data: [fbus, tbus, r, x, b]
    branch_data = np.array([
        [1, 4, 0, 0.0576, 0],
        [4, 5, 0.017, 0.092, 0.158],
        [5, 6, 0.039, 0.17, 0.358],
        [3, 6, 0, 0.0586, 0],
        [6, 7, 0.0119, 0.1008, 0.209],
        [7, 8, 0.0085, 0.072, 0.149],
        [8, 2, 0, 0.0625, 0],
        [8, 9, 0.032, 0.161, 0.306],
        [9, 4, 0.01, 0.085, 0.176]
    ])

    ybus = np.zeros((nbus, nbus), dtype=complex)

    for branch in branch_data:
        fbus = int(branch[0]) - 1  # Convert to 0-indexed
        tbus = int(branch[1]) - 1
        r = branch[2]
        x = branch[3]
        b = branch[4]

        z = r + 1j * x
        if abs(z) < 1e-10:
            raise ValueError(f"Invalid branch impedance (near zero) for branch {fbus+1}-{tbus+1}")
        y = 1 / z
        b_shunt = 1j * b

        # Add to Ybus
        ybus[fbus, fbus] += y + b_shunt / 2
        ybus[tbus, tbus] += y + b_shunt / 2
        ybus[fbus, tbus] -= y
        ybus[tbus, fbus] -= y

    return np.real(ybus), np.imag(ybus)


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
              patience=15, save_dir='checkpoints', criterion=None, model_filename='best_model.pth'):
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        if criterion is None:
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

                checkpoint_path = os.path.join(save_dir, model_filename)
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

    def plot_training_history(self, save_path=None, title='Training History', show=True):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


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
    """Main training function - trains all 6 model combinations"""
    # Configuration
    config = {
        'batch_size': 64,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'patience': 15,
        'data_dir': '../data',
        'save_dir': 'checkpoints'
    }

    print("=" * 60)
    print("AC POWER FLOW NEURAL NETWORK TRAINING")
    print("Training all model/loss combinations")
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

    # Build Ybus matrix for physics-informed loss
    ybus_real, ybus_imag = build_ybus_matrix()

    # Define model types and loss types
    model_names = ['baseline', 'improved', 'physics']
    loss_types = ['mse', 'physics_informed']

    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)

    # Store results for all combinations
    all_results = {}

    # Train all 6 combinations
    for model_name in model_names:
        for loss_type in loss_types:
            print("\n" + "=" * 60)
            print(f"TRAINING: {model_name} model with {loss_type} loss")
            print("=" * 60)

            # Create model
            model = get_model(
                model_name=model_name,
                input_dim=metadata['input_dim'],
                output_dim=metadata['output_dim']
            )

            print(f"Model: {model_name}")
            print(f"Loss: {loss_type}")
            print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            print()

            # Create trainer
            trainer = PowerFlowTrainer(model)

            # Select criterion
            if loss_type == 'mse':
                criterion = nn.MSELoss()
            else:
                criterion = PhysicsInformedLoss(
                    ybus_real=ybus_real,
                    ybus_imag=ybus_imag,
                    nbus=9,
                    slack_bus=0,
                    lambda_mse=1.0,
                    lambda_vm=0.1,
                    lambda_angle=0.1,
                    lambda_power=0.01
                ).to(trainer.device)

            # Define model filename
            model_filename = f'{model_name}_{loss_type}_model.pth'
            history_filename = f'{model_name}_{loss_type}_history.json'
            plot_filename = f'{model_name}_{loss_type}_history.png'

            # Train model
            train_losses, val_losses = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config['num_epochs'],
                learning_rate=config['learning_rate'],
                patience=config['patience'],
                save_dir=config['save_dir'],
                criterion=criterion,
                model_filename=model_filename
            )

            # Save training history with specific filename
            history_path = os.path.join(config['save_dir'], history_filename)
            with open(history_path, 'w') as f:
                json.dump({
                    'model_name': model_name,
                    'loss_type': loss_type,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': trainer.best_val_loss,
                }, f)

            # Plot training history using the trainer's method
            plot_path = os.path.join(config['save_dir'], plot_filename)
            trainer.plot_training_history(
                save_path=plot_path,
                title=f'{model_name} with {loss_type} loss',
                show=False
            )

            # Store results
            final_train_loss = train_losses[-1] if len(train_losses) > 0 else None
            all_results[f'{model_name}_{loss_type}'] = {
                'model_name': model_name,
                'loss_type': loss_type,
                'best_val_loss': trainer.best_val_loss,
                'final_train_loss': final_train_loss,
                'epochs_trained': len(train_losses)
            }

            print(f"Model saved as: {model_filename}")
            print(f"Best validation loss: {trainer.best_val_loss:.6f}")

    # Print summary of all results
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY - ALL 6 COMBINATIONS")
    print("=" * 60)
    print(f"{'Model':<15} {'Loss Type':<20} {'Best Val Loss':<15} {'Epochs':<10}")
    print("-" * 60)
    for key, result in all_results.items():
        print(f"{result['model_name']:<15} {result['loss_type']:<20} "
              f"{result['best_val_loss']:<15.6f} {result['epochs_trained']:<10}")
    print("=" * 60)

    # Save overall results summary
    summary_path = os.path.join(config['save_dir'], 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nTraining summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()