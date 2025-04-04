import os
import torch
import numpy as np
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv

load_dotenv()  # Load environment variables
from PIL import Image
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class GenerationResult:
    image: Image.Image
    video: torch.Tensor
    clip_score: float
    video_path: str = "generated_video.mp4"

class VideoDiffusionModel:
    def __init__(self, use_i2v: bool = True):
        """Initialize diffusion pipelines and CLIP model."""
        self.text2img_pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        self.i2v_pipeline = None
        if use_i2v:
            try:
                self.i2v_pipeline = DiffusionPipeline.from_pretrained(
                    "ali-vilab/i2vgen-xl",
                    token=os.getenv("HUGGINGFACE_TOKEN")
                )
            except Exception as e:
                print(f"Warning: Could not load I2VGenXL pipeline: {e}")
                print("Video generation will be disabled. To enable:")
                print("1. Ensure you have proper HuggingFace authentication")
                print("2. Check if the model is available at ali-vilab/i2vgen-xl")
                print("3. Or provide alternative model path")
                
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            token=os.getenv("HUGGINGFACE_TOKEN"),
            use_fast=True
        )

    def generate_initial_frame(self, prompt: str) -> Image.Image:
        """Generate initial image frame from text prompt."""
        return self.text2img_pipeline(prompt).images[0]

    def generate_video(
        self,
        prompt: str,
        first_frame: Image.Image,
        output_path: str = "generated_video.mp4",
        num_frames: int = 12,  
        chunk_size: int = 4    
    ) -> Optional[torch.Tensor]:
        """Generate video from prompt and first frame with motion control."""
        if not self.i2v_pipeline:
            print("Video generation disabled - I2VGenXL pipeline not loaded")
            return None
            
        try:
            # Generate video frames with error handling
            output = self.i2v_pipeline(
                prompt=prompt,
                image=first_frame,
                num_frames=num_frames,
                decode_chunk_size=chunk_size
            )
            
            # Save as MP4 using provided output path
            self.save_video_frames(output.frames[0], output_path)
            return torch.from_numpy(output.frames[0])
            
        except Exception as e:
            print(f"Video generation failed: {str(e)}")
            print("Falling back to simple frame interpolation")
            return self.fallback_video(first_frame, num_frames, output_path)

    def fallback_video(self, frame: Image.Image, num_frames: int, output_path: str) -> torch.Tensor:
        """Create simple interpolated video when generation fails."""
        try:
            # Convert PIL Image to numpy array and repeat for frames
            frame_array = np.array(frame)
            frames = np.stack([frame_array] * num_frames)
            
            # Save the repeated frames as video
            self.save_video_frames(frames, output_path)
            return torch.from_numpy(frames)
            
        except Exception as e:
            print(f"Fallback video failed: {str(e)}")
            return None

    def save_video_frames(self, frames: np.ndarray, path: str):
        """Save video frames to MP4 file"""
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames.shape[2], frames.shape[3]
        video = cv2.VideoWriter(path, fourcc, 30, (width, height))
        
        for frame in frames:
            frame = (frame.transpose(1, 2, 0) * 255).astype(np.uint8)
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video.release()

    def evaluate_output(
        self,
        image: Image.Image,
        prompt: str
    ) -> float:
        """Evaluate image-prompt alignment using CLIP score."""
        inputs = self.clip_processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True
        )
        outputs = self.clip_model(**inputs)
        return outputs.logits_per_image.item()

    def generate(
        self,
        prompt: str,
        output_path: str = "generated_video.mp4"
    ) -> GenerationResult:
        """Complete generation pipeline from text to video.
        
        Args:
            prompt: Text prompt for video generation
            output_path: Path to save the generated video (default: generated_video.mp4)
            
        Returns:
            GenerationResult containing:
            - Generated image (first frame)
            - Video tensor
            - CLIP score
            - Path to saved video file
        """
        first_frame = self.generate_initial_frame(prompt)
        video = self.generate_video(prompt, first_frame)
        score = self.evaluate_output(first_frame, prompt)
        return GenerationResult(
            image=first_frame,
            video=video if video is not None else torch.zeros(1),  # Dummy tensor if no video
            clip_score=score,
            video_path=output_path
        )
