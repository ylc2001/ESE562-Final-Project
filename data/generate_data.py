import numpy as np
import pandas as pd
from scipy.linalg import inv
import pickle
import os


class PowerFlowDataGenerator:
    """
    Generate training data for AC power flow prediction using ML
    """

    def __init__(self):
        """Initialize the 9-bus system parameters"""
        self.baseMVA = 100.0
        self.nbus = 9
        self.ngen = 3

        # Bus data: [bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin]
        # Type: 1=PQ, 2=PV, 3=Slack
        self.bus_data = np.array([
            [1, 3, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [2, 2, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [3, 2, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [4, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [5, 1, 90, 30, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [6, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [7, 1, 100, 35, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [8, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [9, 1, 125, 50, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9]
        ])

        # Generator data: [bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, ...]
        self.gen_data = np.array([
            [1, 72.3, 27.03, 300, -300, 1.04, 100, 1, 250, 10],
            [2, 163, 6.54, 300, -300, 1.025, 100, 1, 300, 10],
            [3, 85, -10.95, 300, -300, 1.025, 100, 1, 270, 10]
        ])

        # Branch data: [fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status]
        self.branch_data = np.array([
            [1, 4, 0, 0.0576, 0, 250, 250, 250, 0, 0, 1],
            [4, 5, 0.017, 0.092, 0.158, 250, 250, 250, 0, 0, 1],
            [5, 6, 0.039, 0.17, 0.358, 150, 150, 150, 0, 0, 1],
            [3, 6, 0, 0.0586, 0, 300, 300, 300, 0, 0, 1],
            [6, 7, 0.0119, 0.1008, 0.209, 150, 150, 150, 0, 0, 1],
            [7, 8, 0.0085, 0.072, 0.149, 250, 250, 250, 0, 0, 1],
            [8, 2, 0, 0.0625, 0, 250, 250, 250, 0, 0, 1],
            [8, 9, 0.032, 0.161, 0.306, 250, 250, 250, 0, 0, 1],
            [9, 4, 0.01, 0.085, 0.176, 250, 250, 250, 0, 0, 1]
        ])

        # Build Ybus matrix
        self.ybus = self._build_ybus()

        # Identify bus types
        self.bus_types = self.bus_data[:, 1].astype(int)
        self.bus_PV = np.where(self.bus_types == 2)[0]  # PV buses
        self.bus_PQ_all = np.where(self.bus_types == 1)[0]  # All PQ buses

        # Only use PQ buses with non-zero load (buses 5, 7, 9 in 1-indexed, 4, 6, 8 in 0-indexed)
        # This matches the MATLAB code: NN_input = [Pload_random(bus_PQ); Qload_random(bus_PQ)]
        self.bus_PQ = np.array([4, 6, 8])  # Buses with actual loads

        self.bus_slack = np.where(self.bus_types == 3)[0]  # Slack bus
        self.bus_nonslack = np.concatenate([self.bus_PV, self.bus_PQ_all])

        # Original loads (in p.u.)
        self.Pload_base = self.bus_data[:, 2] / self.baseMVA
        self.Qload_base = self.bus_data[:, 3] / self.baseMVA

        # Generator injections (in p.u.)
        self.Pgen = self.gen_data[:, 1] / self.baseMVA
        self.Qgen = self.gen_data[:, 2] / self.baseMVA

        # Build B matrices for fast decoupled power flow
        self._build_B_matrices()

    def _build_ybus(self):
        """Build the admittance matrix (Ybus)"""
        ybus = np.zeros((self.nbus, self.nbus), dtype=complex)

        for branch in self.branch_data:
            fbus = int(branch[0]) - 1  # Convert to 0-indexed
            tbus = int(branch[1]) - 1
            r = branch[2]
            x = branch[3]
            b = branch[4]

            z = r + 1j * x
            y = 1 / z if abs(z) > 1e-10 else 0
            b_shunt = 1j * b

            # Add to Ybus
            ybus[fbus, fbus] += y + b_shunt / 2
            ybus[tbus, tbus] += y + b_shunt / 2
            ybus[fbus, tbus] -= y
            ybus[tbus, fbus] -= y

        return ybus

    def _build_B_matrices(self):
        """Build B' and B'' matrices for fast decoupled load flow"""
        # B' matrix (for angle updates)
        self.B1 = -np.imag(self.ybus[np.ix_(self.bus_nonslack, self.bus_nonslack)])
        self.invB1 = inv(self.B1)

        # B'' matrix (for voltage updates) - uses all PQ buses
        self.B11 = -np.imag(self.ybus[np.ix_(self.bus_PQ_all, self.bus_PQ_all)])
        self.invB11 = inv(self.B11)

    def run_power_flow(self, Pload_random, Qload_random, max_iter=100, tol=1e-8):
        # Net injections at each bus
        Psp = -Pload_random.copy()
        Qsp = -Qload_random.copy()

        # Add generator injections
        gen_buses = self.gen_data[:, 0].astype(int) - 1
        for i, bus in enumerate(gen_buses):
            Psp[bus] += self.Pgen[i]
            Qsp[bus] += self.Qgen[i]

        # Initialize voltage
        Va = np.zeros(self.nbus)
        Vm = np.ones(self.nbus)
        Vbus = Vm * np.exp(1j * Va)

        # Iterative solution
        converged = False
        for iteration in range(max_iter):
            # Calculate power mismatches
            S = Vbus * np.conj(self.ybus @ Vbus)
            Ssp = Psp + 1j * Qsp
            S_mismatch = S - Ssp

            dP = np.real(S_mismatch[self.bus_nonslack])
            dQ = np.imag(S_mismatch[self.bus_PQ_all])

            # Check convergence
            max_mismatch = max(np.max(np.abs(dP)), np.max(np.abs(dQ)))
            if max_mismatch < tol:
                converged = True
                break

            # Update angles
            dVa = -self.invB1 @ dP
            Va[self.bus_nonslack] += dVa

            # Update voltages
            dVm = -self.invB11 @ dQ
            Vm[self.bus_PQ_all] += dVm

            # Update complex voltage
            Vbus = Vm * np.exp(1j * Va)

        return Va, Vm, converged

    def generate_dataset(self, n_samples=10000, load_variation=(0.5, 1.5), seed=42):
        np.random.seed(seed)

        X_list = []
        y_list = []
        failed_samples = 0

        print(f"Generating {n_samples} samples...")
        print(f"Load variation range: {load_variation[0]:.1f}x to {load_variation[1]:.1f}x")

        for i in range(n_samples):
            # Random load scaling factors
            rand_factors = np.random.uniform(load_variation[0], load_variation[1], 2 * self.nbus)

            Pload_random = self.Pload_base * rand_factors[:self.nbus]
            Qload_random = self.Qload_base * rand_factors[self.nbus:]

            # Run power flow
            Va, Vm, converged = self.run_power_flow(Pload_random, Qload_random)

            if not converged:
                failed_samples += 1
                continue

            # Extract input features (only PQ bus loads)
            # Following the MATLAB code: NN_input = [Pload_random(bus_PQ); Qload_random(bus_PQ)]
            X = np.concatenate([Pload_random[self.bus_PQ], Qload_random[self.bus_PQ]])

            # Extract output labels (all bus voltages)
            # Following the MATLAB code: NN_output = [Va; Vm]
            y = np.concatenate([Va, Vm])

            X_list.append(X)
            y_list.append(y)

            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{n_samples} samples ({failed_samples} failed)")

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"\nDataset generation complete!")
        print(f"  Total samples: {len(X)}")
        print(f"  Failed samples: {failed_samples}")
        print(f"  Input shape: {X.shape}")
        print(f"  Output shape: {y.shape}")

        metadata = {
            'n_samples': len(X),
            'failed_samples': failed_samples,
            'load_variation': load_variation,
            'bus_PQ': self.bus_PQ,
            'input_dim': X.shape[1],
            'output_dim': y.shape[1],
            'feature_names': [f'P_load_bus{i + 1}' for i in self.bus_PQ] +
                             [f'Q_load_bus{i + 1}' for i in self.bus_PQ],
            'output_names': [f'Va_bus{i + 1}' for i in range(self.nbus)] +
                            [f'Vm_bus{i + 1}' for i in range(self.nbus)]
        }

        return X, y, metadata

    def save_dataset(self, X, y, metadata, filename='power_flow_dataset.npz'):
        """Save generated dataset to file"""
        save_path = os.path.join(os.path.dirname(__file__), filename)
        np.savez(save_path, X=X, y=y, metadata=metadata)
        print(f"\nDataset saved to: {save_path}")
        return save_path


def main():
    """Main function to generate and save dataset"""
    # Initialize generator
    generator = PowerFlowDataGenerator()

    # Generate training dataset
    X_train, y_train, metadata = generator.generate_dataset(
        n_samples=10000,
        load_variation=(0.5, 1.5),
        seed=42
    )

    # Save dataset
    generator.save_dataset(X_train, y_train, metadata, 'train_dataset.npz')

    # Generate validation dataset with different seed
    X_val, y_val, _ = generator.generate_dataset(
        n_samples=2000,
        load_variation=(0.5, 1.5),
        seed=123
    )

    generator.save_dataset(X_val, y_val, metadata, 'val_dataset.npz')

    # Generate test dataset (potentially out-of-distribution)
    X_test, y_test, _ = generator.generate_dataset(
        n_samples=2000,
        load_variation=(0.3, 1.7),  # Wider range for testing
        seed=456
    )

    generator.save_dataset(X_test, y_test, metadata, 'test_dataset.npz')

    print("\n" + "=" * 60)
    print("DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Training set:   {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set:       {X_test.shape[0]} samples")
    print(f"Input dimension: {X_train.shape[1]} (PQ bus loads)")
    print(f"Output dimension: {y_train.shape[1]} (All bus voltages)")
    print("=" * 60)


if __name__ == "__main__":
    main()