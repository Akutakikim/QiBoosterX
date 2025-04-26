# quantum_config.py

import torch

class QuantumConfig:
    """
    Global configuration class for QiBoosterX.
    Encapsulates all quantum training parameters.
    """

    def __init__(self):
        # Superposition: Number of imagined future states per parameter
        self.superposition_choices = 5

        # Tunneling: Probability to jump across a loss barrier
        self.tunneling_probability = 0.2

        # Entanglement: Frequency of shared state sync across parameter groups
        self.entanglement_groups = 4

        # Collapse: Strategy to choose the final parameter from superpositions
        self.collapse_selection_method = "lowest_loss"

        # Quantum noise: Represents randomness in parameter updates
        self.noise_level = 0.01

        # Loss impact on parameter evolution
        self.loss_scaling = 1.0

        # Training hyperparameters
        self.learning_rate = 0.001
        self.device = self._get_device()

        print(f"[QiBoosterX] Initialized on device: {self.device.upper()}")

    def _check_cuda(self):
        return torch.cuda.is_available()

    def _get_device(self):
        return "cuda" if self._check_cuda() else "cpu"