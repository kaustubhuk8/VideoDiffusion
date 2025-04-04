from epicdiffusion import VideoDiffusionModel
from epicdiffusion.prompt_engine import PromptRefiner
import torch

def main():
    # Initialize models
    print("Initializing models...")
    model = VideoDiffusionModel()
    refiner = PromptRefiner()  # Can pass OpenAI API key if available

    # Example prompt
    prompt = "A futuristic city skyline at sunset with flying cars"

    # Refine prompt using Chain-of-Thought
    print("Refining prompt...")
    elements = refiner.decompose_prompt(prompt)
    refined_prompt = refiner.refine_prompt(elements)
    print(f"Original prompt: {prompt}")
    print(f"Refined prompt: {refined_prompt}")

    # Generate video
    print("Generating video...")
    result = model.generate(refined_prompt)

    # Save results
    result.image.save("first_frame.png")
    if result.video is not None and result.video.numel() > 1:  # Check if real video was generated
        torch.save(result.video, "generated_video.pt")
        print("Saved first_frame.png and generated_video.pt")
    else:
        print("Saved first_frame.png (video generation skipped)")
    print(f"CLIP Score: {result.clip_score:.2f}")

if __name__ == "__main__":
    main()
