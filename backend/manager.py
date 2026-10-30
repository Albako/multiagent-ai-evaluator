import os
import sys
import gc
import time
import resource
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
active_model_metadata: Dict[str, Dict[str, Any]] = {}
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
NODE_ROLE = os.getenv("NODE_ROLE", "worker")
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


def _get_process_rss_mb() -> float:
    """Returns peak resident set size (RSS) in MB, handling Linux vs macOS unit differences."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        # macOS returns ru_maxrss in bytes
        return round(usage / (1024 * 1024), 2)
    # Linux returns ru_maxrss in kilobytes
    return round(usage / 1024, 2)


def _unload_worker_model(worker_id: str) -> None:
    """Safely closes and frees a loaded Llama instance from RAM/VRAM."""
    active_model_metadata.pop(worker_id, None)
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


@app.get("/health")
@app.get("/api/v1/system/status")
def get_system_status():
    """Returns node configuration, active workers, loaded models, and KV cache settings."""
    workers_status: Dict[str, Any] = {}
    with registry_lock:
        for wid, meta in active_model_metadata.items():
            lock = model_locks.get(wid)
            workers_status[wid] = {
                **meta,
                "is_generating": lock.locked() if lock else False,
            }

    return {
        "status": "online",
        "node_role": NODE_ROLE,
        "gpu_offload_supported": bool(llama_cpp.llama_supports_gpu_offload()),
        "process_peak_rss_mb": _get_process_rss_mb(),
        "default_config": {
            "threads": LLAMA_THREADS,
            "threads_batch": LLAMA_THREADS_BATCH,
            "n_batch": N_BATCH,
            "use_mmap": USE_MMAP,
            "flash_attn": FLASH_ATTN,
            "kv_cache_type_k": KV_CACHE_TYPE_K,
            "kv_cache_type_v": KV_CACHE_TYPE_V,
            "override_gpu_layers": OVERRIDE_GPU_LAYERS,
            "override_n_ctx": OVERRIDE_N_CTX,
        },
        "active_workers_count": len(workers_status),
        "active_workers": workers_status,
    }


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
            model_size_mb = round(os.path.getsize(model_path) / (1024 * 1024), 2)

            print(f"Loading {request.model_name} ({model_size_mb} MB) for {request.worker_id} into VRAM...", flush=True)
            print(
                f"Hardware Config -> mmap:{USE_MMAP}, flash_attn:{FLASH_ATTN}, "
                f"kv_cache:{req_k_str}/{req_v_str}, batch:{N_BATCH}, "
                f"threads:{LLAMA_THREADS}, gpu_layers:{safe_gpu_layers}, ctx:{final_n_ctx}",
                flush=True
            )

            load_start = time.monotonic()
            llm = Llama(
            
