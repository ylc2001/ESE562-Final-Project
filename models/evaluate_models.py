import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from mlp_models import get_model


def parse_combo_name(combo_name):
    """Parse combination name into model name and loss type."""
    parts = combo_name.split('_', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid combo name format: {combo_name}")
    return parts[0], parts[1]


def evaluate_single_model(model, X_test, y_test, device):
    """Evaluate a single model and return metrics."""
    model.eval()
    model = model.to(device)
    X_test_dev = X_test.to(device)
    y_test_dev = y_test.to(device)

    with torch.no_grad():
        predictions = model(X_test_dev)

    # Overall metrics
    mse = torch.mean((predictions - y_test_dev) ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - y_test_dev)).item()

    # Separate metrics for angles and magnitudes
    Va_pred = predictions[:, :9].cpu().numpy()
    Vm_pred = predictions[:, 9:].cpu().numpy()
    Va_true = y_test_dev[:, :9].cpu().numpy()
    Vm_true = y_test_dev[:, 9:].cpu().numpy()

    Va_rmse = np.sqrt(np.mean((Va_pred - Va_true)**2))
    Vm_rmse = np.sqrt(np.mean((Vm_pred - Vm_true)**2))

    # Relative errors
    relative_error_Va = np.abs((Va_pred - Va_true) / (Va_true + 1e-8)) * 100
    relative_error_Vm = np.abs((Vm_pred - Vm_true) / Vm_true) * 100

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'Va_rmse': Va_rmse,
        'Vm_rmse': Vm_rmse,
        'Va_relative_error': np.mean(relative_error_Va),
        'Vm_relative_error': np.mean(relative_error_Vm),
        'Va_pred': Va_pred,
        'Vm_pred': Vm_pred,
        'Va_true': Va_true,
        'Vm_true': Vm_true
    }


def main():
    # Load test data
    test_data = np.load('../data/test_dataset.npz', allow_pickle=True)
    X_test = torch.FloatTensor(test_data['X'])
    y_test = torch.FloatTensor(test_data['y'])
    metadata = test_data['metadata'].item()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("MODEL EVALUATION - COMPARING ALL 6 COMBINATIONS")
    print("=" * 80)
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Device: {device}")
    print()

    # Define all 6 combinations
    model_names = ['baseline', 'improved', 'physics']
    loss_types = ['mse', 'physics_informed']

    # Store results for all combinations
    all_results = {}

    # Evaluate each combination
    for model_name in model_names:
        for loss_type in loss_types:
            combo_name = f"{model_name}_{loss_type}"
            model_path = f'checkpoints/{combo_name}_model.pth'

            if not os.path.exists(model_path):
                print(f"WARNING: Model file not found: {model_path}")
                continue

            # Load model
            model = get_model(model_name, input_dim=metadata['input_dim'],
                              output_dim=metadata['output_dim'])
            checkpoint = torch.load(model_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Evaluate
            results = evaluate_single_model(model, X_test, y_test, device)
            all_results[combo_name] = results

    # Print results table
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON TABLE")
    print("=" * 80)

    # Header
    header = f"{'Model':<12} {'Loss Type':<18} {'MSE':<12} {'RMSE':<12} {'MAE':<12}"
    print(header)
    print("-" * 80)

    for combo_name, results in all_results.items():
        model_name, loss_type = parse_combo_name(combo_name)
        print(f"{model_name:<12} {loss_type:<18} {results['mse']:<12.2e} "
              f"{results['rmse']:<12.2e} {results['mae']:<12.2e}")

    # Detailed metrics table
    print("\n" + "=" * 80)
    print("DETAILED METRICS BY OUTPUT TYPE")
    print("=" * 80)

    header2 = (f"{'Model':<12} {'Loss Type':<18} {'Va RMSE':<14} {'Vm RMSE':<14} "
               f"{'Va Rel.Err%':<12} {'Vm Rel.Err%':<12}")
    print(header2)
    print("-" * 80)

    for combo_name, results in all_results.items():
        model_name, loss_type = parse_combo_name(combo_name)
        print(f"{model_name:<12} {loss_type:<18} {results['Va_rmse']:<14.2e} "
              f"{results['Vm_rmse']:<14.2e} {results['Va_relative_error']:<12.2f} "
              f"{results['Vm_relative_error']:<12.2f}")

    # Find best model
    print("\n" + "=" * 80)
    print("BEST MODEL SUMMARY")
    print("=" * 80)

    best_mse_model = min(all_results.items(), key=lambda x: x[1]['mse'])
    best_rmse_model = min(all_results.items(), key=lambda x: x[1]['rmse'])
    best_mae_model = min(all_results.items(), key=lambda x: x[1]['mae'])
    best_va_model = min(all_results.items(), key=lambda x: x[1]['Va_rmse'])
    best_vm_model = min(all_results.items(), key=lambda x: x[1]['Vm_rmse'])

    print(f"Best MSE:           {best_mse_model[0]} ({best_mse_model[1]['mse']:.2e})")
    print(f"Best RMSE:          {best_rmse_model[0]} ({best_rmse_model[1]['rmse']:.2e})")
    print(f"Best MAE:           {best_mae_model[0]} ({best_mae_model[1]['mae']:.2e})")
    print(f"Best Va RMSE:       {best_va_model[0]} ({best_va_model[1]['Va_rmse']:.2e})")
    print(f"Best Vm RMSE:       {best_vm_model[0]} ({best_vm_model[1]['Vm_rmse']:.2e})")

    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Bar chart for MSE comparison
    combo_names = list(all_results.keys())
    mse_values = [all_results[name]['mse'] for name in combo_names]
    rmse_values = [all_results[name]['rmse'] for name in combo_names]
    mae_values = [all_results[name]['mae'] for name in combo_names]
    va_rmse_values = [all_results[name]['Va_rmse'] for name in combo_names]
    vm_rmse_values = [all_results[name]['Vm_rmse'] for name in combo_names]

    # Shorten names for display
    display_names = [name.replace('_', '\n') for name in combo_names]

    # MSE comparison
    bars1 = axes[0, 0].bar(display_names, mse_values, color='steelblue')
    axes[0, 0].set_ylabel('MSE', fontsize=11)
    axes[0, 0].set_title('Mean Squared Error Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=0, labelsize=8)
    axes[0, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # RMSE comparison
    bars2 = axes[0, 1].bar(display_names, rmse_values, color='darkorange')
    axes[0, 1].set_ylabel('RMSE', fontsize=11)
    axes[0, 1].set_title('Root Mean Squared Error Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=0, labelsize=8)
    axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # MAE comparison
    bars3 = axes[0, 2].bar(display_names, mae_values, color='forestgreen')
    axes[0, 2].set_ylabel('MAE', fontsize=11)
    axes[0, 2].set_title('Mean Absolute Error Comparison', fontsize=12, fontweight='bold')
    axes[0, 2].tick_params(axis='x', rotation=0, labelsize=8)
    axes[0, 2].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # Va RMSE comparison
    bars4 = axes[1, 0].bar(display_names, va_rmse_values, color='crimson')
    axes[1, 0].set_ylabel('Va RMSE (rad)', fontsize=11)
    axes[1, 0].set_title('Voltage Angle RMSE Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=0, labelsize=8)
    axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Vm RMSE comparison
    bars5 = axes[1, 1].bar(display_names, vm_rmse_values, color='purple')
    axes[1, 1].set_ylabel('Vm RMSE (p.u.)', fontsize=11)
    axes[1, 1].set_title('Voltage Magnitude RMSE Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=0, labelsize=8)
    axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Grouped bar chart for Va vs Vm RMSE
    x = np.arange(len(combo_names))
    width = 0.35
    bars6a = axes[1, 2].bar(x - width/2, va_rmse_values, width, label='Va RMSE', color='crimson')
    bars6b = axes[1, 2].bar(x + width/2, vm_rmse_values, width, label='Vm RMSE', color='purple')
    axes[1, 2].set_ylabel('RMSE', fontsize=11)
    axes[1, 2].set_title('Va vs Vm RMSE by Model', fontsize=12, fontweight='bold')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(display_names, fontsize=8)
    axes[1, 2].legend()
    axes[1, 2].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('checkpoints/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison chart saved: checkpoints/model_comparison.png")

    # Create prediction vs true value plots for the best model
    best_model_name = best_mse_model[0]
    best_results = best_mse_model[1]

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Voltage angle
    axes2[0].scatter(best_results['Va_true'].flatten(),
                     best_results['Va_pred'].flatten(), alpha=0.3, s=1)
    va_min = best_results['Va_true'].min()
    va_max = best_results['Va_true'].max()
    axes2[0].plot([va_min, va_max], [va_min, va_max], 'r--', lw=2, label='Perfect Prediction')
    axes2[0].set_xlabel('True Voltage Angle (rad)', fontsize=11)
    axes2[0].set_ylabel('Predicted Voltage Angle (rad)', fontsize=11)
    axes2[0].set_title(f'Best Model ({best_model_name})\nVoltage Angles '
                       f'(RMSE={best_results["Va_rmse"]:.2e})', fontsize=12, fontweight='bold')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)

    # Voltage magnitude
    axes2[1].scatter(best_results['Vm_true'].flatten(),
                     best_results['Vm_pred'].flatten(), alpha=0.3, s=1)
    vm_min = best_results['Vm_true'].min()
    vm_max = best_results['Vm_true'].max()
    axes2[1].plot([vm_min, vm_max], [vm_min, vm_max], 'r--', lw=2, label='Perfect Prediction')
    axes2[1].set_xlabel('True Voltage Magnitude (p.u.)', fontsize=11)
    axes2[1].set_ylabel('Predicted Voltage Magnitude (p.u.)', fontsize=11)
    axes2[1].set_title(f'Best Model ({best_model_name})\nVoltage Magnitudes '
                       f'(RMSE={best_results["Vm_rmse"]:.2e})', fontsize=12, fontweight='bold')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('checkpoints/best_model_predictions.png', dpi=300, bbox_inches='tight')
    print("Best model prediction plot saved: checkpoints/best_model_predictions.png")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()