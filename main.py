from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import datetime
import re

from sd3_infer import load_models


# Lifespan setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the model here
    app.state.inferencer = load_models(model_folder="/home/wanerevinita/sd3.5/models")
    print("✅ Model loaded and ready")
    yield
    # (Optional) Cleanup on shutdown
    print("👋 Server shutting down")


# Create app with lifespan
app = FastAPI(lifespan=lifespan)

# (Optional) Allow CORS for a front-end UI later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schema
class GenerateRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 50
    cfg: float = 5.0
    sampler: str = "dpmpp_2m"
    seed: int = 42
    out_dir: str = "./outputs"
    init_image: str = None
    controlnet_cond_image: str = None
    denoise: float = 0.5
    seed_type: str = "rand"


# Image generation endpoint
@app.post("/generate")
async def generate(req: GenerateRequest):
    sanitized_prompt = re.sub(r"[^\w\-\.]", "_", req.prompt)
    timestamp = datetime.datetime.now().strftime("_%Y-%m-%dT%H-%M-%S")
    output_path = os.path.join(req.out_dir, sanitized_prompt + timestamp)
    print(output_path, flush=True)
    try:
        os.makedirs(output_path, exist_ok=True)
    except Exception as e:
        print(f"❌ Failed to create directory: {e}", flush=True)

    app.state.inferencer.gen_image(
        [req.prompt],
        req.width,
        req.height,
        req.steps,
        req.cfg,
        req.sampler,
        req.seed,
        req.seed_type,
        req.out_dir,
        req.controlnet_cond_image,
        req.init_image,
        req.denoise,
    )

    return JSONResponse(
        content={"message": "✅ Image generated", "output_dir": output_path}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
