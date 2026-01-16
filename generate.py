import torch
from diffusers import ZImagePipeline

print("Starting...")
print("Note: Z-Image is a 6B parameter model. This may take time and require ~16GB RAM.")

# Use CPU-only mode with sequential loading for Mac M4
print("Loading model from Hugging Face...")
print("Using CPU mode with sequential loading to minimize memory usage...")

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.float32,   # float32 for CPU
    low_cpu_mem_usage=True,      # Load weights sequentially
    device_map=None,              # Don't use automatic device mapping
)

# Move to CPU explicitly
pipe = pipe.to("cpu")
print("Model loaded successfully on CPU!")
prompt = "A cinematic portrait of a Japanese young woman in traditional pink Kimono, night scene"

print("Generating image (this will take several minutes on CPU)...")
image = pipe(
    prompt=prompt,
    height=512,
    width=512,
    num_inference_steps=9,  # This results in 8 DiT forwards
    guidance_scale=0.0,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]

image.save("zimage_output1.png")
print("✅ Done! Image saved as zimage_output1.png")
