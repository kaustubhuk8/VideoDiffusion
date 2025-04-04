import torch
from typing import Union, List
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPScorer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize CLIP model for text-video alignment scoring."""
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def calculate_score(
        self,
        frames: Union[torch.Tensor, List[Image.Image]],
        text_prompt: str,
        frame_weights: Optional[List[float]] = None
    ) -> float:
        """Calculate CLIP similarity score between video frames and text prompt."""
        if isinstance(frames, list):
            # Process list of PIL Images
            inputs = self.processor(
                text=[text_prompt],
                images=frames,
                return_tensors="pt",
                padding=True
            )
        else:
            # Process tensor of frames
            inputs = self.processor(
                text=[text_prompt],
                images=[Image.fromarray(frame.numpy()) for frame in frames],
                return_tensors="pt",
                padding=True
            )

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        frame_scores = outputs.logits_per_image.flatten()
        
        if frame_weights:
            # Apply weighted average if weights provided
            weights = torch.tensor(frame_weights, dtype=torch.float32)
            weights = weights / weights.sum()
            return (frame_scores * weights).sum().item()
        
        return frame_scores.mean().item()

    def batch_score(
        self,
        videos: List[Union[torch.Tensor, List[Image.Image]]],
        text_prompts: List[str]
    ) -> List[float]:
        """Calculate CLIP scores for multiple video-text pairs."""
        return [
            self.calculate_score(video, prompt)
            for video, prompt in zip(videos, text_prompts)
        ]
