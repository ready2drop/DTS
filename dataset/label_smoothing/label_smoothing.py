from typing import Dict

import torch
import torch.nn.functional as F


class LabelSmoothing:
    def __init__(self, alpha: float = 0.1, **kwargs: Dict):
        self.alpha = alpha

    def one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        num_classes = labels.size(1)
        return F.one_hot(labels.long(), num_classes=num_classes).squeeze(1).permute(0, 4, 1, 2, 3).float()

    def __call__(self, labels: torch.Tensor) -> torch.Tensor:
        return labels

class KNeighborLabelSmoothing3D(LabelSmoothing):
    def __init__(
        self,
        alpha: float = 0.1,
        k: int = 3,
        method: str = "rational",
        num_classes: int = 14,
        smoothing_order: int = 1,
        lambda_decay: float = 1.0,
        epsilon: float = 1e-6,
        **kwargs: Dict
    ):
        super().__init__(alpha=alpha)
        self.k = k
        self.method = method
        self.num_classes = num_classes
        self.smoothing_order = smoothing_order
        self.lambda_decay = lambda_decay
        self.epsilon = epsilon

    def rational(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / (x.pow(self.smoothing_order) + self.epsilon)

    def exponential_decay(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.exp(-self.lambda_decay * x)

    def damped_sine(self, x: torch.Tensor, lambda_decay: float = 0.05, omega: float = 0.1, phi: float = 0) -> torch.Tensor:
        return torch.exp(-lambda_decay * x) * torch.sin(omega * x + phi)

    def distance_equation(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "rational":
            return self.rational(x) * self.alpha
        elif self.method == "exponential":
            return self.exponential_decay(x) * self.alpha
        elif self.method == "damped_sine":
            return self.damped_sine(x) * self.alpha
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")

    def run(self, labels: torch.Tensor) -> torch.Tensor:
        """K-neighbor label smoothing"""
        labels = F.one_hot(labels.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        org = labels.squeeze(0)
        B, C, W, H, D = labels.shape

        # Pre-compute all indices
        indices = torch.stack(torch.meshgrid(
            torch.arange(W),
            torch.arange(H),
            torch.arange(D),
            indexing='ij'
        ), dim=-1).float().to(labels.device) # Shape: [W, H, D, 3]

        # Initialize tensor to hold centroids
        centroids = torch.zeros(self.num_classes, 3, device=labels.device)

        # Calculate class-wise centroids
        for i in range(self.num_classes):
            # Extract the i-th channel for class i
            class_mask = labels[:, i, :, :, :]

            # Ensure the mask is 4D and of boolean type
            class_mask = class_mask.squeeze(0).bool()  # Reduce first dimension and convert to bool

            # Ensure indices and class_mask have compatible shapes
            if class_mask.shape != indices.shape[:-1]:
                raise ValueError(f"Shape mismatch: class_mask has shape {class_mask.shape}, but indices has shape {indices.shape}")

            masked_indices = indices[class_mask]
            if masked_indices.numel() > 0:  # Check if there are any elements
                centroid = masked_indices.mean(dim=0)
                centroids[i] = centroid

        # Expand centroids to match broadcasting dimensions
        centroids = centroids[:, None, None, None, None, :]

        # Calculate distances with correct broadcasting
        distances = torch.norm(indices[None, None, :, :, :, :] - centroids, dim=-1)

        labels = self.distance_equation(distances.squeeze(1))
        labels = torch.abs(org - labels)

        return labels

    def __call__(self, labels: torch.Tensor) -> torch.Tensor:
        return self.run(labels)


class LabelSmoothingHub:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, alpha: float, **kwargs) -> LabelSmoothing:
        if self.name == "k-nls":
            return KNeighborLabelSmoothing3D(alpha=alpha, **kwargs)
        else:
            return LabelSmoothing(alpha=alpha, **kwargs)