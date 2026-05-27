import runpod
import torch
import diffusers
import base64
import io
import os
from models import RequestPayload, ResponsePayload
from sdnq import SDNQConfig
from sdnq.loader import apply_sdnq_options_to_model

MODEL_ID = "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32"
CACHE_DIR = "/runpod-volume/huggingface-cache/hub"

os.environ["HF_HOME"] = "/runpod-volume/huggingface-cache"
os.environ["HF_HUB_CACHE"] = CACHE_DIR

print("Loading pipeline...")
pipe = diffusers.Flux2KleinPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
    local_files_only=False
)

print("Applying SDNQ quantization optimizations...")
pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
pipe.enable_model_cpu_offload()

print("Model ready.")

def handler(job):
    inp = RequestPayload.model_validate(job["input"])
    prompt = inp.prompt
    height = inp.height
    width = inp.width

    generator = torch.Generator(device="cuda").manual_seed(0)

    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=generator
    ).images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return ResponsePayload(image=img_base64).model_dump()

runpod.serverless.start({"handler": handler})