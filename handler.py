import runpod
import torch
import diffusers
import base64
import io
import os
from PIL import Image
from models import RequestPayload, ResponsePayload

MODEL_ID = "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32"
CACHE_DIR = "/runpod-volume/huggingface-cache/hub"

os.environ["HF_HOME"] = "/runpod-volume/huggingface-cache"
os.environ["HF_HUB_CACHE"] = CACHE_DIR

def prepare_image(base64_str: str):
    img = Image.open(io.BytesIO(base64.b64decode(base64_str)))
    img = img.convert("RGB")
    w, h = img.size
    w = min(w - (w % 16), 2560)
    h = min(h - (h % 16), 2560)
    if w != img.size[0] or h != img.size[1]:
        img = img.resize((w, h))
    return img

print("Loading pipeline...")
pipe = diffusers.Flux2KleinPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
    local_files_only=False
)

pipe.enable_model_cpu_offload()

print("Model ready.")

def handler(job):
    inp = RequestPayload.model_validate(job["input"])
    prompt = inp.prompt
    height = inp.height
    width = inp.width

    generator = torch.Generator(device="cuda").manual_seed(0)

    pipe_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
        "generator": generator
    }

    images = []
    if inp.input_image:
        images.append(prepare_image(inp.input_image))
    if inp.reference_images:
        for ref in inp.reference_images:
            images.append(prepare_image(ref))
    if images:
        pipe_kwargs["image"] = images

    image = pipe(**pipe_kwargs).images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return ResponsePayload(image=img_base64).model_dump()

runpod.serverless.start({"handler": handler})