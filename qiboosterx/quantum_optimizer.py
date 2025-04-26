# quantum_optimizer.py

import torch
import numpy as np

class QuantumOptimizer:
    def __init__(self, model_params, lr=0.01, config=None):
        """
        Quantum-inspired optimizer that adds noise-based perturbations to model parameters.
        
        Args:
            model_params (iterable): Model parameters to optimize.
            lr (float): Learning rate (step size).
            config (object): Optional configuration object with noise_level attribute.
        """
        self.model_params = model_params
        self.lr = lr
        self.config = config or type("Config", (), {"noise_level": 0.05})()

    def quantum_adjustment(self, parameter):
        """
        Simulates a quantum-inspired parameter adjustment using noise.
        
        Args:
            parameter (Tensor): The parameter tensor to adjust.
        
        Returns:
            Tensor: Adjusted parameter tensor.
        """
        quantum_shift = self.lr * torch.randn_like(parameter) * self.config.noise_level
        return parameter - quantum_shift

    def step(self):
        """
        Apply quantum-inspired optimization step to all model parameters.
        """
        with torch.no_grad():
            for param in self.model_params:
                if param.grad is not None:
                    updated_param = self.quantum_adjustment(param)
                    param.copy_(updated_param)

    def zero_grad(self):
        """
        Reset gradients for all model parameters (standard PyTorch compatibility).
        """
        for param in self.model_params:
            if param.grad is not None:
                param.grad.zero_()