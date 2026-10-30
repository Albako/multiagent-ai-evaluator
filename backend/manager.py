import os
import gc
import traceback
import threading
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import llama_cpp
from llama_cpp import Llama

app = FastAPI(title="Dynamic Model Manager")

# Program manages which models to load into the memory
# Key is worker_id (worker1 or judge)
# Value is instantiated Llama object
active_models: Dict[str, Llama] = {}
model_locks: Dict[str, threading.Lock] = {}
registry_lock = threading.Lock()

# Map string names to ggml_type integer constants required by llama-cpp-python
KV_TYPE_MAP: Dict[str, int] = {
    "f32": llama_cpp.GGML_TYPE_F32,
    "f16": llama_cpp.GGML_TYPE_F16,
    "bf16": getattr(llama_cpp, "GGML_TYPE_BF16", 30),
    "q8_0": llama_cpp.GGML_TYPE_Q8_0,
    "q5_1": llama_cpp.GGML_TYPE_Q5_1,
    "q5_0": llama_cpp.GGML_TYPE_Q5_0,
    "q4_1": llama_cpp.GGML_TYPE_Q4_1,
    "q4_0": llama_cpp.GGML_TYPE_Q4_0,
}

# Hardware & runtime configuration from environment variables
LLAMA_THREADS = int(os.getenv("LLAMA_THREADS", 8))
LLAMA_THREADS_BATCH = int(os.getenv("LLAMA_THREADS_BATCH", 8))
N_BATCH = int(os.getenv("N_BATCH", 512))
USE_MMAP = os.getenv("USE_MMAP", "true").lower() == "true"
FLASH_ATTN = os.getenv("FLASH_ATTN", "true").lower() == "true"
KV_CACHE_TYPE_K = os.getenv("KV_CACHE_TYPE_K", "q8_0").lower()
KV_CACHE_TYPE_V = os.getenv("KV_CACHE_TYPE_V", "q8_0").lower()

# Optional overrides (None if not set or set to "0" for N_CTX)
RAW_OVERRIDE_GPU = os.getenv("OVERRIDE_GPU_LAYERS")
OVERRIDE_GPU_LAYERS = int(RAW_OVERRIDE_GPU) if RAW_OVERRIDE_GPU is not None else None
OVERRIDE_N_CTX = int(os.getenv("OVERRIDE_N_CTX", 0))


class LoadModelRequest(BaseModel):
    worker_id: str
    model_name: str
    n_gpu_layers: int = -1  # -1 = all layers to GPU; 0 = CPU only
    n_ctx: int = 8192
    type_k: Optional[str] = None  # e.g., "f16", "q8_0", "q4_0" (defaults to ENV / q8_0)
    type_v: Optional[str] = None  # e.g., "f16", "q8_0", "q4_0" (defaults to ENV / q8_0)


class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    json_schema: Optional[Dict[str, Any]] = None  # Triggers GBNF grammar enforcement in llama.cpp


def _unload_worker_model(worker_id: str) -> None:
    """Safely closes and frees a loaded Llama instance from RAM/VRAM."""
    if worker_id in active_models:
        print(f"Unloading existing model for {worker_id}...", flush=True)
        llm = active_models.pop(worker_id)
        try:
            if hasattr(llm, "close"):
                llm.close()
        except Exception as e:
            print(f"Warning during llm.close() for {worker_id}: {e}", flush=True)
        del llm
        gc.collect()


@app.post("/api/v1/system/load_model")
def load_model(request: LoadModelRequest):
    MODELS_DIR = os.getenv("MODELS_DIR", "/models")
    model_path = os.path.join(MODELS_DIR, request.model_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found at {model_path}")

    # Resolve KV cache types (request overrides ENV, ENV defaults to q8_0)
    req_k_str = (request.type_k or KV_CACHE_TYPE_K).lower()
    req_v_str = (request.type_v or KV_CACHE_TYPE_V).lower()

    if req_k_str not in KV_TYPE_MAP or req_v_str not in KV_TYPE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid KV cache type. Allowed values: {list(KV_TYPE_MAP.keys())}"
        )

    # Guard: llama.cpp requires flash_attn=True when V-cache is quantized
    if not FLASH_ATTN and req_v_str != "f16":
        print(
            f"WARNING: Quantized V-cache ({req_v_str}) requires FLASH_ATTN=true. Falling back type_v to f16.",
            flush=True
        )
        req_v_str = "f16"

    ggml_type_k = KV_TYPE_MAP[req_k_str]
    ggml_type_v = KV_TYPE_MAP[req_v_str]

    with registry_lock:
        if request.worker_id not in model_locks:
            model_locks[request.worker_id] = threading.Lock()
        worker_lock = model_locks[request.worker_id]

    with worker_lock:
        _unload_worker_model(request.worker_id)

        try:
            # Apply OVERRIDE_GPU_LAYERS if set to non-zero (supports -1 for full offload)
            if OVERRIDE_GPU_LAYERS is not None and OVERRIDE_GPU_LAYERS != 0:
                safe_gpu_layers = OVERRIDE_GPU_LAYERS
            else:
                safe_gpu_layers = request.n_gpu_layers

            final_n_ctx = OVERRIDE_N_CTX if OVERRIDE_N_CTX > 0 else request.n_ctx

            print(f"Loading {request.model_name} for {request.worker_id} into VRAM...", flush=True)
            print(
                f"Hardware Config -> mmap:{USE_MMAP}, flash_attn:{FLASH_ATTN}, "
                f"kv_cache:{req_k_str}/{req_v_str}, batch:{N_BATCH}, "
                f"threads:{LLAMA_THREADS}, gpu_layers:{safe_gpu_layers}, ctx:{final_n_ctx}",
                flush=True
            )

            llm = Llama(
                model_path=model_path,
                n_gpu_layers=safe_gpu_layers,
                n_ctx=final_n_ctx,
                n_batch=N_BATCH,
                n_threads=LLAMA_THREADS,
                n_threads_batch=LLAMA_THREADS_BATCH,
                use_mmap=USE_MMAP,
                flash_attn=FLASH_ATTN,
                type_k=ggml_type_k,
                type_v=ggml_type_v,
                verbose=True
            )
            active_models[request.worker_id] = llm

            return {
                "status": "success",
                "message": f"Successfully loaded {request.model_name} for {request.worker_id} (KV: {req_k_str}/{req_v_str})"
            }
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
def clear_all_models():
    with registry_lock:
        worker_ids = list(active_models.keys())
        for wid in worker_ids:
            lock = model_locks.get(wid)
            if lock:
                with lock:
                    _unload_worker_model(wid)
            else:
                _unload_worker_model(wid)
        model_locks.clear()
        gc.collect()

    return {"status": "success", "message": "All models have been unloaded from RAM/VRAM."}
    
