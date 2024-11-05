from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_aer import AerSimulator
from qiskit_algorithms.utils import algorithm_globals

class QSVM_Kernel:
    def __init__(self, feature_dimension=2, reps=2, seed=42, feature_map_type="ZZ"):
        self.feature_dimension = feature_dimension
        self.reps = reps
        self.seed = seed
        self.feature_map_type = feature_map_type

        # Select feature map based on the specified type
        if self.feature_map_type == "ZZ":
            self.feature_map = ZZFeatureMap(
                feature_dimension=self.feature_dimension, 
                reps=self.reps
            )
        elif self.feature_map_type == "Pauli":
            self.feature_map = PauliFeatureMap(
                feature_dimension=self.feature_dimension,
                reps=self.reps,
                paulis=['X', 'Y', 'Z']
            )
        else:
            raise ValueError("Invalid feature_map_type specified. Choose 'ZZ' or 'Pauli'.")

        # Set random seed for reproducibility
        algorithm_globals.random_seed = seed

        self.quantum_kernel = FidelityQuantumKernel(
            feature_map=self.feature_map,
            fidelity=None,  # fidelity is set to None to use the default fidelity function
            enforce_psd=True,  # Ensures positive semi-definite kernel matrix
            evaluate_duplicates="off_diagonal"
        )

    def get_kernel_matrix(self, X):
        # Calculate the kernel matrix for the input data X
        return self.quantum_kernel.evaluate(x_vec=X)
