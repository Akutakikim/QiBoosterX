# quantum_particle.py

import torch
import random

class QuantumParticle:
    """
    Represents a model parameter behaving like a quantum particle, capable of:
    - Superposition: holding multiple possible states
    - Tunneling: escaping local minima
    - Entanglement: syncing with other particles
    - Collapse: resolving into the best possible state
    """

    def __init__(self, param, config):
        self.param = param
        self.config = config
        self.device = getattr(config, "device", torch.device("cpu"))
        self.superpositions = []

    def superpose(self):
        """
        Generate multiple possible future states (superpositions) for this parameter.
        """
        self.superpositions.clear()
        for _ in range(self.config.superposition_choices):
            noise = torch.randn_like(self.param).to(self.device) * self.config.noise_level
            self.superpositions.append(self.param + noise)

    def tunnel(self):
        """
        Simulate quantum tunneling by randomly shifting the parameter.
        """
        if random.random() < self.config.tunneling_probability:
            tunnel_noise = torch.randn_like(self.param).to(self.device) * (self.config.noise_level * 10)
            self.param.data.add_(tunnel_noise)

    def collapse(self, losses):
        """
        Collapse the particle to the best superposition state based on loss values.
        """
        if not self.superpositions or not losses:
            return

        losses_tensor = torch.tensor(losses, device=self.device)
        best_idx = torch.argmin(losses_tensor).item()
        self.param.data.copy_(self.superpositions[best_idx].data)

    def entangle(self, other_particle):
        """
        Simulate quantum entanglement by syncing states with another particle.
        """
        if random.randint(0, self.config.entanglement_groups) == 0:
            avg = (self.param + other_particle.param) / 2
            self.param.data.copy_(avg.data)
            other_particle.param.data.copy_(avg.data)