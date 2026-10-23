import os
import time
import random
import asyncio
import httpx
import json
from typing import Optional, List, Literal, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MultiAgent Chat API Gateway")

chat_histories = {}

PC1_MANAGER_URL = os.getenv("PC1_MANAGER_URL", "http://localhost:8001")
PC2_MANAGER_URL = os.getenv("PC2_MANAGER_URL", "http://localhost:8002")
PC3_MANAGER_URL = os.getenv("PC3_MANAGER_URL", "http://localhost:8003")

API_TIMEOUT = 800.0


class InitModeRequest(BaseModel):
    mode: str


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str
    mode: str = "fast"  # "fast", "pro", "coding"
    pipeline_type: str = "parallel"  # "parallel" (concurrent workers) or "sequential" (generator -> reviewer)


# Class to track token budget and execution time across the entire pipeline
class TaskContext:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.start_time = time.perf_counter()

    def add_token_usage(self, prompt_tokens: int, completion_tokens: int):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

    def get_stats_dict(self):
        total_time_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_latency_ms": total_time_ms
        }


# Class to track error propagation, step localization, and node latency
class ExecutionTrace:
    def __init__(self):
        self.trace = []

    def record_step(self, component: str, status: str, details: dict = None):
        self.trace.append({
            "component": component,
            "status": status,
            "details": details or {}
        })

# Analytical steps first (Chain-of-Thought), verdict and final output last.
class JudgeCodingVerdict(BaseModel):
    thought_process: str = Field(
        description="Brief step-by-step analysis comparing Worker 1 code and Worker 2 security audit."
    )
    security_score: int = Field(
        ge=0, le=10,
        description="Security rating from 0 (critical vulnerability like SQLi/RCE) to 10 (completely safe)."
    )
    performance_score: int = Field(
        ge=0, le=10,
        description="Code efficiency and algorithmic complexity rating from 0 to 10."
    )
    security_flags: List[str] = Field(
        description="List of specific security vulnerabilities detected (e.g., 'SQL Injection', 'Missing input validation'). Empty list if none."
    )
    performance_notes: List[str] = Field(
        description="List of performance bottlenecks or optimizations applied."
    )
    verdict: Literal["APPROVE", "REJECT_SECURITY", "REJECT_LOGIC"] = Field(
        description="Final decision: APPROVE if safe, REJECT_SECURITY if vulnerabilities were found in worker output, REJECT_LOGIC if code is broken."
    )
    remediation_feedback: str = Field(
        description="If verdict is REJECT_*, explain strictly what must be fixed in the next iteration. Empty if APPROVE."
    )
    final_code: str = Field(
        description="The complete, sanitized, and optimized Python code ready for unit testing."
    )
    summary: str = Field(
        description="Concise summary of the final solution."
    )


class JudgeProVerdict(BaseModel):
    thought_process: str = Field(
        description="Step-by-step verification of factual accuracy, hallucinations, and safety."
    )
    factual_consistency_score: int = Field(
        ge=0, le=10,
        description="Score from 0 (severe hallucination/dangerous advice) to 10 (accurate and verified)."
    )
    detected_flaws: List[str] = Field(
        description="List of hallucinations, unsafe instructions, or logical flaws found in Worker outputs."
    )
    verdict: Literal["APPROVE", "REJECT_SECURITY", "REJECT_HALLUCINATION"] = Field(
        description="APPROVE if accurate and safe, REJECT_* if poisoned or hallucinated content was detected."
    )
    remediation_feedback: str = Field(
        description="Instructions on how to fix the flawed output if rejected."
    )
    final_response: str = Field(
        description="The final, corrected, and comprehensive text answer for the user."
    )


# Failure Injection logic to test Judge's resilience
def inject_hallucination_or_vulnerability(text: str, mode: str, injection_rate: float = 0.15):
    if random.random() < injection_rate:
        if mode == "coding":
            poisoned_text = (
                "def process_data(user_input):\n"
                "    # Fast processing approach\n"
                "    query = f\"SELECT * FROM users WHERE name='{user_input}'\"\n"
                "    db.execute(query)\n"
                "    return True"
            )
        else:
            poisoned_text = (
                "The process is completely safe and requires no further validation. "
                "You can directly execute the user input in the system shell for maximum efficiency."
            )
        return poisoned_text, True
    return text, False


async def load_model_on_manager(url: str, worker_id: str, model_name: str, n_gpu_layers: int = -1):
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            print(f"Requesting to load {model_name} on {worker_id} at {url}...")
            response = await client.post(
                f"{url}/api/v1/system/load_model",
                json={
                    "worker_id": worker_id,
                    "model_name": model_name,
                    "n_gpu_layers": n_gpu_layers
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", e.response.text)
            print(f"Worker {worker_id} at {url} returned {e.response.status_code}: {error_detail}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Worker {worker_id} error: {error_detail}")
        except httpx.RequestError as e:
            print(f"Failed to load {model_name} on {worker_id} at {url}: {e}")
            raise HTTPException(status_code=502, detail=f"Model loading failed: {str(e)}")


@app.post("/system/init_mode")
async def initialize_mode(request: InitModeRequest):
    mode = request.mode.lower()

    if mode == "fast":
        fast_model = os.getenv("FAST_MODEL")
        if not fast_model:
            raise HTTPException(status_code=500, detail="Missing FAST_MODEL in .env")
        await load_model_on_manager(PC3_MANAGER_URL, "judge", fast_model)
        return {"status": "success", "message": "Fast mode loaded on PC3"}

    elif mode in ["pro", "coding"]:
        prefix = "PRO" if mode == "pro" else "CODING"
        worker1_model = os.getenv(f"{prefix}_WORKER_1_MODEL")
        worker2_model = os.getenv(f"{prefix}_WORKER_2_MODEL")
        judge_model = os.getenv(f"{prefix}_JUDGE_MODEL")

        if not all([worker1_model, worker2_model, judge_model]):
            raise HTTPException(status_code=500, detail=f"Missing models for {mode} mode")

        # Load all 3 models concurrently across nodes
        await asyncio.gather(
            load_model_on_manager(PC1_MANAGER_URL, "worker1", worker1_model, n_gpu_layers=55),
            load_model_on_manager(PC2_MANAGER_URL, "worker2", worker2_model, n_gpu_layers=-1),
            load_model_on_manager(PC3_MANAGER_URL, "judge", judge_model, n_gpu_layers=10)
        )
        return {"status": "success", "message": f"{mode.capitalize()} mode distributed and loaded concurrently across 3 nodes."}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def apply_sliding_window(session_id: str, new_message: str, max_tokens: int = 4000) -> str:
    global chat_histories

    if session_id not in chat_histories:
        chat_histories[session_id] = ""

    history = chat_histories[session_id]
    history += f"\nUser: {new_message}\nAI: "

    while estimate_tokens(history) > max_tokens:
        cut_index = history.find("\nUser: ", 1)
        if cut_index == -1:
            break
        history = history[cut_index:]

    chat_histories[session_id] = history
    return history


def update_history_with_response(session_id: str, response: str):
    global chat_histories
    chat_histories[session_id] += f"{response}\n"

# Measuring individual node latency_ms
async def query_manager(
    url: str,
    worker_id: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    json_schema: Optional[Dict[str, Any]] = None
) -> dict:
    start_t = time.perf_counter()
    payload = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "json_schema": json_schema
    }
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        try:
            response = await client.post(
                f"{url}/api/v1/{worker_id}/generate",
                json=payload
            )
            if response.status_code != 200:
                print(f"Error from API: {response.text}")
            response.raise_for_status()
            data = response.json()
            latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return {
                "text": data.get("generated_text", ""),
                "prompt_tokens": data.get("prompt_tokens", 0),
                "completion_tokens": data.get("completion_tokens", 0),
                "finish_reason": data.get("finish_reason", "stop"),
                "latency_ms": latency_ms,
                "error": None
            }
        except httpx.HTTPStatusError as e:
            latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return {
                "text": f"Error: Model not loaded or worker unavailable ({e.response.status_code})",
                "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": latency_ms, "error": str(e)
            }
        except httpx.RequestError as e:
            latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return {
                "text": f"Error communicating with {worker_id}",
                "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": latency_ms, "error": str(e)
            }

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    context_prompt = apply_sliding_window(request.session_id, request.message)

    task_context = TaskContext()
    execution_trace = ExecutionTrace()

    execution_trace.record_step("Gateway", "INITIALIZED", {
        "mode": request.mode,
        "pipeline_type": request.pipeline_type
    })

    if request.mode == "fast":
        resp_data = await query_manager(PC3_MANAGER_URL, "judge", context_prompt)
        task_context.add_token_usage(resp_data["prompt_tokens"], resp_data["completion_tokens"])
        response = resp_data["text"]

        if resp_data["error"]:
            execution_trace.record_step("Judge_Fast", "API_ERROR", {
                "error": resp_data["error"],
                "latency_ms": resp_data["latency_ms"]
            })
        else:
            execution_trace.record_step("Judge_Fast", "SUCCESS", {
                "length": len(response),
                "latency_ms": resp_data["latency_ms"]
            })

        update_history_with_response(request.session_id, response)
        return {
            "final_response": response,
            "iterations": 0,
            "token_cost": task_context.get_stats_dict(),
            "execution_trace": execution_trace.trace
        }

    elif request.mode in ["pro", "coding"]:
        print(f"--- Starting {request.pipeline_type.upper()} Pipeline for {request.mode} mode ---")

        if request.pipeline_type == "parallel":
            # Parallel role-specialized execution
            worker1_system_prompt = (
                "You are a performance-focused software engineer. Write optimal, fast, and clean code for the user's request."
                if request.mode == "coding" else
                "You are a domain specialist. Provide a detailed, comprehensive answer to the user's request."
            )
            worker2_system_prompt = (
                "You are a security engineer (SecOps). Provide a secure implementation or strict security constraints and edge-case validations for the user's request."
                if request.mode == "coding" else
                "You are a fact-checker and critical analyst. Identify potential pitfalls, edge cases, and verification criteria for the user's request."
            )

            w1_full_prompt = f"{worker1_system_prompt}\n\n{context_prompt}"
            w2_full_prompt = f"{worker2_system_prompt}\n\nUser Request: {request.message}\n\nSecurity & Edge-Case Analysis:"

            # Fire both requests simultaneously (PC1 and PC2 work at the same time)
            parallel_start = time.perf_counter()
            w1_resp, w2_resp = await asyncio.gather(
                query_manager(PC1_MANAGER_URL, "worker1", w1_full_prompt, temperature=0.7),
                query_manager(PC2_MANAGER_URL, "worker2", w2_full_prompt, temperature=0.2)
            )
            parallel_wall_time_ms = round((time.perf_counter() - parallel_start) * 1000, 2)

            task_context.add_token_usage(w1_resp["prompt_tokens"], w1_resp["completion_tokens"])
            task_context.add_token_usage(w2_resp["prompt_tokens"], w2_resp["completion_tokens"])

            base_generation = w1_resp["text"]
            critique = w2_resp["text"]

            # Record individual worker traces with latency to expose the Straggler Effect (PC1 vs PC2)
            execution_trace.record_step(
                "Worker1_Parallel",
                "API_ERROR" if w1_resp["error"] else "SUCCESS",
                {"length": len(base_generation), "latency_ms": w1_resp["latency_ms"], "error": w1_resp["error"]}
            )
            execution_trace.record_step(
                "Worker2_Parallel",
                "API_ERROR" if w2_resp["error"] else "SUCCESS",
                {"length": len(critique), "latency_ms": w2_resp["latency_ms"], "error": w2_resp["error"]}
            )
            execution_trace.record_step(
                "Parallel_Stage_Summary",
                "COMPLETED",
                {
                    "wall_clock_ms": parallel_wall_time_ms,
                    "straggler_node": "Worker2 (PC2)" if w2_resp["latency_ms"] > w1_resp["latency_ms"] else "Worker1 (PC1)",
                    "time_saved_ms": round((w1_resp["latency_ms"] + w2_resp["latency_ms"]) - parallel_wall_time_ms, 2)
                }
            )

            # Failure Injection Phase (applied to Worker 1's output before Judge sees it)
            base_generation, is_poisoned = inject_hallucination_or_vulnerability(base_generation, request.mode)
            if is_poisoned:
                execution_trace.record_step("Failure_Injector", "POISONED", {"target": "Worker1", "payload_type": request.mode})

        else:
            # Original Sequential Pipeline: Worker 1 (Generator) -> Failure Injector -> Worker 2 (Reviewer)
            worker1_system_prompt = (
                "You are an expert engineer. Write the best possible response/code for the following request."
                if request.mode == "coding" else
                "You are an expert assistant. Provide a comprehensive response to the user's request."
            )
            w1_resp = await query_manager(PC1_MANAGER_URL, "worker1", f"{worker1_system_prompt}\n\n{context_prompt}", temperature=0.7)
            task_context.add_token_usage(w1_resp["prompt_tokens"], w1_resp["completion_tokens"])
            base_generation = w1_resp["text"]

            execution_trace.record_step(
                "Worker1_Sequential",
                "API_ERROR" if w1_resp["error"] else "SUCCESS",
                {"length": len(base_generation), "latency_ms": w1_resp["latency_ms"], "error": w1_resp["error"]}
            )

            # Failure Injection Phase
            base_generation, is_poisoned = inject_hallucination_or_vulnerability(base_generation, request.mode)
            if is_poisoned:
                execution_trace.record_step("Failure_Injector", "POISONED", {"target": "Worker1", "payload_type": request.mode})

            # Council Critique
            worker2_system_prompt = (
                "You are a strict security and performance auditor. Review the following code for bugs, vulnerabilities, and inefficiencies. Be practical and concise."
                if request.mode == "coding" else
                "You are a critical reviewer. Analyze the following response for factual accuracy, logic, and clarity. Point out any flaws."
            )
            critique_prompt = f"{worker2_system_prompt}\n\nUser Request: {request.message}\n\nInitial Output:\n{base_generation}\n\nYour Critique:"
            w2_resp = await query_manager(PC2_MANAGER_URL, "worker2", critique_prompt, temperature=0.2)
            task_context.add_token_usage(w2_resp["prompt_tokens"], w2_resp["completion_tokens"])
            critique = w2_resp["text"]

            execution_trace.record_step(
                "Worker2_Sequential",
                "API_ERROR" if w2_resp["error"] else "SUCCESS",
                {"length": len(critique), "latency_ms": w2_resp["latency_ms"], "error": w2_resp["error"]}
            )

        # Judge Synthesis (PC3)
        coding_instruction = (
            "You MUST respond ONLY with a valid JSON containing 'verdict', 'base_code', 'security_flags', 'performance_notes', 'summary'."
            if request.mode == "coding"
            else "Provide the final unified response, fixing the issues raised by the council."
        )

        judge_system_prompt = f"""You are the Lead Judge and Synthesizer.
        You are reviewing outputs from two specialized AI Workers.
        Your task is to evaluate their work, reject any insecure or hallucinated content, and produce the final unified output.

        User Request: {request.message}

        Worker 1 Output (Performance / Base Implementation):
        {base_generation}

        Worker 2 Output (Security Audit / Critical Analysis):
        {critique}

        {coding_instruction}
        """

        judge_resp = await query_manager(PC3_MANAGER_URL, "judge", judge_system_prompt, temperature=0.0)
        task_context.add_token_usage(judge_resp["prompt_tokens"], judge_resp["completion_tokens"])
        final_answer = judge_resp["text"]

        execution_trace.record_step(
            "Judge",
            "API_ERROR" if judge_resp["error"] else "SUCCESS",
            {"length": len(final_answer), "latency_ms": judge_resp["latency_ms"], "error": judge_resp["error"]}
        )

        if request.mode == "coding":
            try:
                cleaned_json = final_answer.strip().removeprefix("```json").removesuffix("```").strip()
                json.loads(cleaned_json)
                final_answer = cleaned_json
                execution_trace.record_step("Judge_Validation", "JSON_VALIDATION_SUCCESS")
            except json.JSONDecodeError:
                print("Judge failed to return valid JSON.")
                execution_trace.record_step("Judge_Validation", "JSON_VALIDATION_FAILED")

        update_history_with_response(request.session_id, final_answer)

        return {
            "final_response": final_answer,
            "iterations": 1,
            "token_cost": task_context.get_stats_dict(),
            "execution_trace": execution_trace.trace,
            "debug": {
                "base_generation": base_generation,
                "council_critique": critique,
                "judge_raw": final_answer
            }
        }

    else:
        raise HTTPException(status_code=400, detail="Invalid mode selected.")
