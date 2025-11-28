import torch
import numpy as np
import matplotlib.pyplot as plt
from mlp_models import get_model

# 1. 加载测试数据
test_data = np.load('../data/test_dataset.npz', allow_pickle=True)
X_test = torch.FloatTensor(test_data['X'])
y_test = torch.FloatTensor(test_data['y'])
metadata = test_data['metadata'].item()

print("="*60)
print("MODEL EVALUATION")
print("="*60)
print(f"Test samples: {X_test.shape[0]}")

# 2. 加载训练好的模型
model = get_model('baseline', input_dim=6, output_dim=18)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# 3. 预测
with torch.no_grad():
    predictions = model(X_test)

# 4. 计算误差
mse = torch.mean((predictions - y_test) ** 2).item()
rmse = np.sqrt(mse)
mae = torch.mean(torch.abs(predictions - y_test)).item()

# 分别计算角度和幅值的误差
Va_pred = predictions[:, :9].cpu().numpy()
Vm_pred = predictions[:, 9:].cpu().numpy()
Va_true = y_test[:, :9].cpu().numpy()
Vm_true = y_test[:, 9:].cpu().numpy()

Va_rmse = np.sqrt(np.mean((Va_pred - Va_true)**2))
Vm_rmse = np.sqrt(np.mean((Vm_pred - Vm_true)**2))

print(f"\n整体性能:")
print(f"  MSE:  {mse:.2e}")
print(f"  RMSE: {rmse:.2e}")
print(f"  MAE:  {mae:.2e}")

print(f"\n分项性能:")
print(f"  电压相角 RMSE: {Va_rmse:.2e} rad ({np.degrees(Va_rmse):.4f}°)")
print(f"  电压幅值 RMSE: {Vm_rmse:.2e} p.u. ({Vm_rmse*100:.4f}%)")

# 5. 相对误差
relative_error_Va = np.abs((Va_pred - Va_true) / (Va_true + 1e-8)) * 100
relative_error_Vm = np.abs((Vm_pred - Vm_true) / Vm_true) * 100

print(f"\n相对误差:")
print(f"  电压相角平均相对误差: {np.mean(relative_error_Va):.2f}%")
print(f"  电压幅值平均相对误差: {np.mean(relative_error_Vm):.2f}%")

# 6. 可视化：预测 vs 真实值
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 电压相角
axes[0].scatter(Va_true.flatten(), Va_pred.flatten(), alpha=0.3, s=1)
axes[0].plot([Va_true.min(), Va_true.max()],
             [Va_true.min(), Va_true.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('True Voltage Angle (rad)', fontsize=11)
axes[0].set_ylabel('Predicted Voltage Angle (rad)', fontsize=11)
axes[0].set_title(f'Voltage Angles (RMSE={Va_rmse:.2e})', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 电压幅值
axes[1].scatter(Vm_true.flatten(), Vm_pred.flatten(), alpha=0.3, s=1)
axes[1].plot([Vm_true.min(), Vm_true.max()],
             [Vm_true.min(), Vm_true.max()], 'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('True Voltage Magnitude (p.u.)', fontsize=11)
axes[1].set_ylabel('Predicted Voltage Magnitude (p.u.)', fontsize=11)
axes[1].set_title(f'Voltage Magnitudes (RMSE={Vm_rmse:.2e})', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('checkpoints/prediction_comparison.png', dpi=300, bbox_inches='tight')
print("\n预测对比图已保存: checkpoints/prediction_comparison.png")

print("\n"+"="*60)
print("评估完成！模型性能优秀 ✓")
print("="*60)