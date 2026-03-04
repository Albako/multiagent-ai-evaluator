import os
import gc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="Dynamic Model Manager")

# Program manages which models to load into the memory
# Key is worker_id (worker1 or judge)
# Value is instantiated Llama object
active_models = {}

class LoadModelRequest(BaseModel):
    worker_id: str
    model_name: str
    n_gpu_layers: int = -1
    n_ctx: int = 8192

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7

@app.post("/api/v1/system/load_model")
async def load_model(request: LoadModelRequest):
    global active_models

    # checks if a model is already loaded
    if request.worker_id in active_models:
        print(f"Unloading existing model for {request.worker_id}...")
        del active_models[request.worker_id]
        gc.collect()

    model_path = os.path.join("/models", request.model_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Page...I mean model not found at {model_path}")
    try:
        print(f"Loading {request.model_name} for {request.worker_id} into VRAM...")
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=request.n_gpu_layers,
            n_ctx=request.n_ctx,
            verbose=False
        )
        active_models[request.worker_id] = llm
        return {"status": "success", "message": f"Successfully loaded {request.model_name} for {request.worker_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/api/v1/{worker_id}/generate")
async def generate_text(worker_id: str, request: GenerateRequest):
    if worker_id not in active_models:
        raise HTTPException(status_code=400, detail=f"No active model loaded for worker ID: {worker_id}")

    llm = active_models[worker_id]

    try:
        response = llm(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return {"worker_id": worker_id, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.post("/api/v1/system/clear_all")
async def clear_all_models():
    global active_models

    active_models.clear()
    gc.collect()

    return {"status": "success", "message": "All models have been unloaded from VRAM."}
