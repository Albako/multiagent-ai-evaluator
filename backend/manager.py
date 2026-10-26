import os
import gc
import traceback
import threading
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="Dynamic Model Manager")

# Program manages which models to load into the memory
# Key is worker_id (worker1 or judge)
# Value is instantiated Llama object
active_models: Dict[str, Llama] = {}
model_locks: Dict[str, threading.Lock] = {}

# Getting number of CPU cores
LLAMA_THREADS = int(os.getenv("LLAMA_THREADS", 8))
LLAMA_THREADS_BATCH = int(os.getenv("LLAMA_THREADS_BATCH", 8))
N_BATCH = int(os.getenv("N_BATCH", 512))
USE_MMAP = os.getenv("USE_MMAP", "true").lower() == "true"
FLASH_ATTN = os.getenv("FLASH_ATTN", "true").lower() == "true"
OVERRIDE_GPU_LAYERS = int(os.getenv("OVERRIDE_GPU_LAYERS", 0))
OVERRIDE_N_CTX = int(os.getenv("OVERRIDE_N_CTX", 0))

class LoadModelRequest(BaseModel):
    worker_id: str
    model_name: str
    n_gpu_layers: int = -1 # -1 = all layers to GPU; 0 = CPU only 
    n_ctx: int = 8192

class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    json_schema: Optional[Dict[str, Any]] = None # Triggers GBNF grammar enforcement in llama.cpp

@app.post("/api/v1/system/load_model")
async def load_model(request: LoadModelRequest):
    global active_models, model_locks

    # checks if a model is already loaded
    if request.worker_id in active_models:
        print(f"Unloading existing model for {request.worker_id}...")
        del active_models[request.worker_id]
        gc.collect()

    MODELS_DIR = os.getenv("MODELS_DIR", "/models")
    model_path = os.path.join(MODELS_DIR, request.model_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Page...I mean model not found at {model_path}")
    try:
        safe_gpu_layers = (
            OVERRIDE_GPU_LAYERS
            if (request.n_gpu_layers == -1 and OVERRIDE_GPU_LAYERS > 0)
            else request.n_gpu_layers
        )
        final_n_ctx = OVERRIDE_N_CTX if OVERRIDE_N_CTX > 0 else request.n_ctx
        
        print(f"Loading {request.model_name} for {request.worker_id} into VRAM...")
        print(f"Hardware Config -> mmap:{USE_MMAP}, flash_attn:{FLASH_ATTN}, batch:{N_BATCH}, threads:{LLAMA_THREADS}")
        
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=safe_gpu_layers,
            n_ctx=final_n_ctx,
            n_batch=N_BATCH,
            n_threads=LLAMA_THREADS,
            n_threads_batch=LLAMA_THREADS_BATCH,
            use_mmap=USE_MMAP,
            flash_attn=FLASH_ATTN,
            verbose=True
        )
        active_models[request.worker_id] = llm
        if request.worker_id not in model_locks:
            model_locks[request.worker_id] = threading.Lock()
            
        return {"status": "success", "message": f"Successfully loaded {request.model_name} for {request.worker_id}"}
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        print(error_msg, flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)
        
@app.post("/api/v1/{worker_id}/generate")
def generate_text(worker_id: str, request: GenerateRequest):
    if worker_id not in active_models:
        raise HTTPException(status_code=400, detail=f"No active model loaded for worker ID: {worker_id}")

    llm = active_models[worker_id]
    lock = model_locks[worker_id]

    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": request.prompt})

    # Configure GBNF structured output if json_schema is provided
    response_format = None
    effective_repeat_penalty = 1.15
    if request.json_schema:
        response_format = {
            "type": "json_object",
            "schema": request.json_schema
        }
        effective_repeat_penalty = 1.02

    try:
        with lock:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                repeat_penalty=effective_repeat_penalty,
                response_format=response_format
            )

        usage = response.get("usage", {})
        p_tokens = usage.get("prompt_tokens", 0)
        c_tokens = usage.get("completion_tokens", 0)
        finish_reason = response["choices"][0].get("finish_reason", "stop")

        return {
            "worker_id": worker_id,
            "generated_text": response["choices"][0]["message"]["content"].strip(),
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
            "finish_reason": finish_reason
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.post("/api/v1/system/clear_all")
async def clear_all_models():
    global active_models, model_locks

    active_models.clear()
    model_locks.clear()
    gc.collect()

    return {"status": "success", "message": "All models have been unloaded from RAM/VRAM."}
