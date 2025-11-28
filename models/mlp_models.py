import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineMLP(nn.Module):
    def __init__(self, input_dim=6, output_dim=18, hidden_dims=[64, 128, 64]):
        super(BaselineMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)


class ImprovedMLP(nn.Module):
    def __init__(self, input_dim=6, output_dim=18, hidden_dims=[128, 256, 128],
                 dropout_rate=0.2, use_batchnorm=True):
        super(ImprovedMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_batchnorm = use_batchnorm

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(nn.ReLU())

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)


class PhysicsInformedMLP(nn.Module):
    def __init__(self, input_dim=6, output_dim=18, hidden_dims=[128, 256, 256, 128],
                 dropout_rate=0.1):
        super(PhysicsInformedMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Shared feature extraction
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Separate heads for angles and magnitudes
        # This allows the network to learn different patterns for each
        self.angle_head = nn.Linear(prev_dim, 9)  # Voltage angles
        self.magnitude_head = nn.Linear(prev_dim, 9)  # Voltage magnitudes

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)

        # Predict angles and magnitudes separately
        angles = self.angle_head(features)
        magnitudes = self.magnitude_head(features)

        # Concatenate outputs
        output = torch.cat([angles, magnitudes], dim=1)

        return output


def get_model(model_name='baseline', input_dim=6, output_dim=18, **kwargs):
    models = {
        'baseline': BaselineMLP,
        'improved': ImprovedMLP,
        'physics': PhysicsInformedMLP
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    model = models[model_name](input_dim=input_dim, output_dim=output_dim, **kwargs)

    return model


if __name__ == "__main__":
    """Test model architectures"""
    print("Testing model architectures...\n")

    batch_size = 32
    input_dim = 6
    output_dim = 18

    # Create dummy input
    x = torch.randn(batch_size, input_dim)

    # Test baseline model
    print("1. Baseline MLP:")
    model_baseline = get_model('baseline', input_dim, output_dim)
    output = model_baseline(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_baseline.parameters()):,}")
    print()

    # Test improved model
    print("2. Improved MLP:")
    model_improved = get_model('improved', input_dim, output_dim)
    output = model_improved(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_improved.parameters()):,}")
    print()

    # Test physics-informed model
    print("3. Physics-Informed MLP:")
    model_physics = get_model('physics', input_dim, output_dim)
    output = model_physics(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_physics.parameters()):,}")
    print()

    print("All models working correctly! ✓")