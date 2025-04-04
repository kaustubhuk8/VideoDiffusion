import torch
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline, I2VGenXLPipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load Stable Diffusion & I2VGenXL pipelines
text2img_pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
i2v_pipeline = I2VGenXLPipeline.from_pretrained("stabilityai/i2vgen-xl")

# Load CLIP model for text-image similarity
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to generate an initial image from text
def generate_initial_frame(prompt):
    image = text2img_pipeline(prompt).images[0]
    return image

# Function to enhance motion and structure consistency
def generate_video(prompt, first_frame):
    return i2v_pipeline(prompt=prompt, image=first_frame).videos[0]

# Function to evaluate generated output
def evaluate_output(image, prompt):
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    return outputs.logits_per_image.item()

# Example Usage
prompt = "A futuristic city skyline at sunset with flying cars"
first_frame = generate_initial_frame(prompt)
video = generate_video(prompt, first_frame)
score = evaluate_output(first_frame, prompt)

print(f"CLIP Similarity Score: {score}")
