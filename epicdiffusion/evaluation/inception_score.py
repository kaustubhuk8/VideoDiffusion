import torch
import numpy as np
from typing import Union, List
from torchvision.models import inception_v3
from torchvision.models import Inception_V3_Weights
from scipy.stats import entropy

class InceptionScorer:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize Inception Score calculator."""
        self.device = device
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model = self.model.to(device)
        self.model.eval()
        
    def calculate_score(
        self,
        videos: Union[List[torch.Tensor], torch.Tensor],
        splits: int = 10
    ) -> float:
        """Calculate Inception Score for videos by averaging frame scores."""
        if isinstance(videos, list):
            videos = torch.stack(videos)
            
        # Process each frame through inception model
        preds = []
        for video in videos:
            frame_preds = []
            for frame in video:
                frame = frame.to(self.device).unsqueeze(0)
                with torch.no_grad():
                    pred = torch.nn.functional.softmax(self.model(frame), dim=1)
                frame_preds.append(pred.cpu().numpy())
            preds.append(np.mean(frame_preds, axis=0))
            
        preds = np.array(preds)
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
            
        return np.mean(scores), np.std(scores)

    def batch_score(
        self,
        video_batches: List[Union[List[torch.Tensor], torch.Tensor]]
    ) -> List[float]:
        """Calculate Inception Scores for multiple video batches."""
        return [self.calculate_score(batch)[0] for batch in video_batches]
