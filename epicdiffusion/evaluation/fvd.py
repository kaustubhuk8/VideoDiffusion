import torch
import numpy as np
from typing import List, Union
from scipy.linalg import sqrtm
from torchvision.models.video import r3d_18
from torchvision.models.video import R3D_18_Weights

class FVDCalculator:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize FVD calculator with I3D features extractor."""
        self.device = device
        self.model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.model = self.model.to(device)
        self.model.eval()
        
    def calculate_fvd(
        self,
        real_videos: Union[List[torch.Tensor], torch.Tensor],
        generated_videos: Union[List[torch.Tensor], torch.Tensor]
    ) -> float:
        """Calculate Frechet Video Distance between real and generated videos."""
        real_features = self._extract_features(real_videos)
        gen_features = self._extract_features(generated_videos)
        
        # Calculate mean and covariance statistics
        mu_real, sigma_real = self._calculate_stats(real_features)
        mu_gen, sigma_gen = self._calculate_stats(gen_features)
        
        # Compute FVD
        diff = mu_real - mu_gen
        covmean = sqrtm(sigma_real.dot(sigma_gen))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        return diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)

    def _extract_features(self, videos: Union[List[torch.Tensor], torch.Tensor]) -> np.ndarray:
        """Extract video features using I3D model."""
        if isinstance(videos, list):
            videos = torch.stack(videos)
            
        videos = videos.to(self.device)
        with torch.no_grad():
            features = self.model(videos)
        return features.cpu().numpy()

    def _calculate_stats(self, features: np.ndarray) -> tuple:
        """Calculate mean and covariance of features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
