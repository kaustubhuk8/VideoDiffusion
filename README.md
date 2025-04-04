# EpicDiffusion - Advanced Video Generation Framework

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch Version](https://img.shields.io/badge/pytorch-2.0+-red.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid text-to-video pipeline combining Stable Diffusion and I2VGenXL with advanced features for storytelling applications.

## Key Features

- **Hybrid Diffusion Pipeline**: Combines Stable Diffusion for initial frame generation with I2VGenXL for video synthesis
- **Chain-of-Thought Prompt Refinement**: Multi-stage LLM-based prompt decomposition for better narrative alignment
- **Motion Synthesis**: AnimateDiff with MotionLoRA for fine-grained motion control
- **Dual-Conditioning**: Combines first-frame image inputs with text prompts for improved consistency
- **Comprehensive Evaluation**: CLIPScore, FVD-UMT, and Inception Score metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kaustubhuk8/VideoDiffusion.git
cd EpicDiffusion
```

2. Set up Python environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Configure authentication:
```bash
# Install environment package
pip install python-dotenv

# Setup configuration
nano .env  # Add your credentials

# Required:
HUGGINGFACE_TOKEN=your_token_here

# Optional (for enhanced prompt refinement):
# OPENAI_API_KEY=your_openai_key_here
```

4. Authenticate models:
```bash
huggingface-cli login
# Accept model terms at:
# - https://huggingface.co/CompVis/stable-diffusion-v1-4
# - https://huggingface.co/ali-vilab/i2vgen-xl
```

## Usage

```python
from epicdiffusion import VideoDiffusionModel
from epicdiffusion.prompt_engine import PromptRefiner

# Initialize models
model = VideoDiffusionModel()
refiner = PromptRefiner()

# Refine prompt using Chain-of-Thought
elements = refiner.decompose_prompt("A futuristic city skyline at sunset with flying cars")
refined_prompt = refiner.refine_prompt(elements)

# Generate video
result = model.generate(refined_prompt)
result.video.save("output.mp4")
```

## Project Structure

```
epicdiffusion/
├── core.py               # Main diffusion pipeline
├── prompt_engine.py      # LLM-based prompt refinement
├── motion/               # Motion synthesis components
│   ├── animate_diff.py
│   └── motion_lora.py
├── evaluation/           # Benchmarking metrics
│   ├── clip_score.py
│   ├── fvd.py
│   └── inception_score.py
```

## Benchmarks

| Metric          | Value |
|-----------------|-------|
| CLIPScore       | 0.85  |
| FVD-UMT         | 45.2  |
| Inception Score | 8.2   |

## Troubleshooting

### Authentication Errors
If you get authentication errors for HuggingFace models:
1. Make sure you're logged in:
```bash
huggingface-cli login
```
2. Accept the model terms at:
   - https://huggingface.co/CompVis/stable-diffusion-v1-4
   - https://huggingface.co/ali-vilab/i2vgen-xl
   - https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2 (for motion adapter)

### Performance Issues
For better performance:
```bash
pip install accelerate
```

### Video Generation Not Working
If video generation is disabled:
1. Check if the I2VGenXL model is available
2. Try alternative models like "cerspense/zeroscope_v2_576w"

## License

MIT License - See [LICENSE](LICENSE) for details.
