"""
Custom SB3 feature extractor for the Dict observation space.

Replaces the default MultiInputPolicy MLP branches with:
  • 1-D CNN for the "lidar"  key  (360 range readings  → 32 features)
  • Small MLP for the "vector" key (7 scalars           → 32 features)
  • Concatenation                  (64-dim output)

CNN architecture
────────────────
  Input  (B, 1, 360)
  Conv1d( 1,  8, k=5, s=3, p=2) → ReLU → (B,  8, 120)
  Conv1d( 8, 16, k=5, s=3, p=2) → ReLU → (B, 16,  40)
  Conv1d(16, 32, k=3, s=2, p=1) → ReLU → (B, 32,  20)
  AdaptiveAvgPool1d(4)                  → (B, 32,   4)
  Flatten                               → (B, 128)
  Linear(128, 32)                → ReLU → (B,  32)

The AdaptiveAvgPool makes the architecture robust to lidar resolution
changes — if 360 rays is later halved or doubled, only the pool changes.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LidarCNNExtractor(BaseFeaturesExtractor):
    """
    1-D CNN for lidar + MLP for vector observation, outputs a
    (lidar_features + vector_features)-dim feature vector to the PPO head.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        lidar_features: int  = 32,
        vector_features: int = 32,
    ) -> None:
        features_dim = lidar_features + vector_features
        super().__init__(observation_space, features_dim=features_dim)

        n_lidar  = observation_space["lidar"].shape[0]   # 360
        n_vector = observation_space["vector"].shape[0]  # 7

        # ── Lidar CNN ──────────────────────────────────────────────────────
        self._lidar_cnn = nn.Sequential(
            nn.Conv1d(1,  8,  kernel_size=5, stride=3, padding=2),
            nn.ReLU(),
            nn.Conv1d(8,  16, kernel_size=5, stride=3, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),   # output always (B, 32, 4) regardless of n_lidar
            nn.Flatten(),              # → (B, 128)
        )

        # Compute flattened CNN output dim without hard-coding it
        with torch.no_grad():
            _cnn_out = self._lidar_cnn(torch.zeros(1, 1, n_lidar)).shape[1]

        self._lidar_head = nn.Sequential(
            nn.Linear(_cnn_out, lidar_features),
            nn.ReLU(),
        )

        # ── Vector MLP ────────────────────────────────────────────────────
        self._vector_net = nn.Sequential(
            nn.Linear(n_vector, vector_features),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # lidar: (B, 360) → add channel dim → (B, 1, 360)
        lidar_in = observations["lidar"].unsqueeze(1)
        lf = self._lidar_head(self._lidar_cnn(lidar_in))

        # vector: (B, 7)
        vf = self._vector_net(observations["vector"])

        return torch.cat([lf, vf], dim=1)   # (B, 64)
