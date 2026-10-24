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

# Node URLs:
# PC1 (8001): Worker 1 - Code/Response Generator
# PC2 (8002): Worker 2 - SecOps Auditor
# PC3 (8003): Worker 3 - QA & Edge-Case Auditor
# PC4 (8004): Judge    - Lead Synthesizer
PC1_MANAGER_URL = os.getenv("PC1_MANAGER_URL", "http://localhost:8001")
PC2_MANAGER_URL = os.getenv("PC2_MANAGER_URL", "http://localhost:8002")
PC3_MANAGER_URL = os.getenv("PC3_MANAGER_URL", "http://localhost:8003")
PC4_MANAGER_URL = os.getenv("PC4_MANAGER_URL", "http://localhost:8004")

API_TIMEOUT = 800.0


class InitModeRequest(BaseModel):
    mode: str


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str
    mode: str = "fast"  # "fast", "pro", "coding"
    pipeline_type: str = "parallel"  # "parallel" (all 3 workers concurrent) or "sequential" (W1 -> W2+W3 concurrent audit)
    max_retries: int = 1  # How many times Worker1 can retry after REJECT before Judge Takeover


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
        description="Brief step-by-step analysis comparing Worker 1 code, Worker 2 security audit, and Worker 3 QA edge-cases."
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
        description="List of performance bottlenecks, edge-case bugs, or optimizations applied."
    )
    verdict: Literal["APPROVE", "REJECT_SECURITY", "REJECT_LOGIC"] = Field(
        description="Final decision: APPROVE if safe, REJECT_SECURITY if vulnerabilities were found in worker output, REJECT_LOGIC if code is broken."
    )
    remediation_feedback: str = Field(
        description="If verdict is REJECT_*, explain strictly what must be fixed by Worker 1. Empty if APPROVE."
    )
    final_code: str = Field(
        description="The complete, sanitized, and optimized Python code ready for unit testing."
    )
    summary: str = Field(
        description="Concise summary of the final solution."
    )


class JudgeProVerdict(BaseModel):
    thought_process: str = Field(
        description="Step-by-step verification of factual accuracy, hallucinations, and safety across all 3 workers."
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
        await load_model_on_manager(PC4_MANAGER_URL, "judge", fast_model)
        return {"status": "success", "message": "Fast mode loaded on PC4 (Judge node)"}

    elif mode in ["pro", "coding"]:
        prefix = "PRO" if mode == "pro" else "CODING"
        worker1_model = os.getenv(f"{prefix}_WORKER_1_MODEL")
        worker2_model = os.getenv(f"{prefix}_WORKER_2_MODEL")
        worker3_model = os.getenv(f"{prefix}_WORKER_3_MODEL")
        judge_model = os.getenv(f"{prefix}_JUDGE_MODEL")

        if not all([worker1_model, worker2_model, worker3_model, judge_model]):
            raise HTTPException(status_code=500, detail=f"Missing models for {mode} mode in .env")

        # Load all 4 models concurrently across PC1, PC2, PC3, and PC4
        await asyncio.gather(
            load_model_on_manager(PC1_MANAGER_URL, "worker1", worker1_model, n_gpu_layers=55),
            load_model_on_manager(PC2_MANAGER_URL, "worker2", worker2_model, n_gpu_layers=-1),
            load_model_on_manager(PC3_MANAGER_URL, "worker3", worker3_model, n_gpu_layers=-1),
            load_model_on_manager(PC4_MANAGER_URL, "judge", judge_model, n_gpu_layers=10)
        )
        return {"status": "success", "message": f"{mode.capitalize()} mode distributed and loaded concurrently across 4 nodes."}

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


# Measuring individual node latency_ms and passing system_prompt / json_schema
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
        "pipeline_type": request.pipeline_type,
        "max_retries": request.max_retries
    })

    if request.mode == "fast":
        resp_data = await query_manager(
            PC4_MANAGER_URL,
            "judge",
            prompt=context_prompt,
            system_prompt="You are a helpful and fast AI assistant."
        )
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
            "resolution_status": "FAST_SINGLE_AGENT",
            "iterations": 0,
            "token_cost": task_context.get_stats_dict(),
            "execution_trace": execution_trace.trace
        }

    elif request.mode in ["pro", "coding"]:
        print(f"--- Starting {request.pipeline_type.upper()} Pipeline for {request.mode} mode ---")

        max_allowed_retries = max(0, request.max_retries)
        remediation_feedback = None
        base_generation = ""
        critique = ""
        final_answer = ""
        parsed_verdict_dict = {}
        resolution_status = "UNKNOWN"
        iterations_used = 0

        # Role definitions across the 3 Workers:
        # Worker 1: Primary Code/Content Generator
        worker1_sys = (
            "You are a performance-focused software engineer. Write optimal, fast, and clean Python code for the user's request."
            if request.mode == "coding" else
            "You are a domain specialist. Provide a detailed, comprehensive answer to the user's request."
        )
        # Worker 2: Security & Vulnerability Auditor (SecOps)
        worker2_parallel_sys = (
            "You are a security engineer (SecOps). Provide strict security constraints, input validation rules, and safe coding patterns for the user's request."
            if request.mode == "coding" else
            "You are a safety and fact-checking analyst. Identify potential security risks, misinformation hazards, and verification criteria."
        )
        worker2_review_sys = (
            "You are a strict security auditor (SecOps). Review the provided code for SQLi, RCE, injection vulnerabilities, and unsafe functions. Be concise."
            if request.mode == "coding" else
            "You are a strict fact-checker. Analyze the provided response for hallucinations, dangerous advice, and factual errors."
        )
        # Worker 3: QA, Logic & Edge-Case Auditor
        worker3_parallel_sys = (
            "You are a QA Lead and software architect. Identify boundary conditions, edge cases, algorithmic bottlenecks, and type-safety requirements for the user's request."
            if request.mode == "coding" else
            "You are a critical logic reviewer. Identify logical gaps, missing context, and structural improvements needed to answer the user's request."
        )
        worker3_review_sys = (
            "You are a QA Lead and performance reviewer. Review the provided code for logical bugs, unhandled exceptions, edge cases, and algorithmic complexity."
            if request.mode == "coding" else
            "You are a logic and clarity auditor. Point out any logical contradictions, omissions, or structural flaws in the provided response."
        )

        for iteration in range(max_allowed_retries + 1):
            iterations_used = iteration + 1
            is_last_attempt = (iteration == max_allowed_retries)

            # --- PHASE 1: WORKER EXECUTION (Initial vs Retry) ---
            if iteration == 0:
                if request.pipeline_type == "parallel":
                    w2_prompt = f"User Request: {request.message}\n\nProvide Security (SecOps) Constraints:"
                    w3_prompt = f"User Request: {request.message}\n\nProvide QA & Edge-Case Constraints:"

                    # Fire all 3 workers concurrently (PC1, PC2, PC3)
                    parallel_start = time.perf_counter()
                    w1_resp, w2_resp, w3_resp = await asyncio.gather(
                        query_manager(PC1_MANAGER_URL, "worker1", prompt=context_prompt, system_prompt=worker1_sys, temperature=0.7),
                        query_manager(PC2_MANAGER_URL, "worker2", prompt=w2_prompt, system_prompt=worker2_parallel_sys, temperature=0.2),
                        query_manager(PC3_MANAGER_URL, "worker3", prompt=w3_prompt, system_prompt=worker3_parallel_sys, temperature=0.3)
                    )
                    parallel_wall_time_ms = round((time.perf_counter() - parallel_start) * 1000, 2)

                    task_context.add_token_usage(w1_resp["prompt_tokens"], w1_resp["completion_tokens"])
                    task_context.add_token_usage(w2_resp["prompt_tokens"], w2_resp["completion_tokens"])
                    task_context.add_token_usage(w3_resp["prompt_tokens"], w3_resp["completion_tokens"])

                    base_generation = w1_resp["text"]
                    critique = (
                        f"[Worker 2 - SecOps Audit (PC2 Jetson)]:\n{w2_resp['text']}\n\n"
                        f"[Worker 3 - QA & Edge-Case Audit (PC3 M1 Pro)]:\n{w3_resp['text']}"
                    )

                    execution_trace.record_step(
                        "Worker1_Parallel_PC1",
                        "API_ERROR" if w1_resp["error"] else "SUCCESS",
                        {"iteration": iteration, "length": len(base_generation), "latency_ms": w1_resp["latency_ms"], "error": w1_resp["error"]}
                    )
                    execution_trace.record_step(
                        "Worker2_Parallel_PC2",
                        "API_ERROR" if w2_resp["error"] else "SUCCESS",
                        {"iteration": iteration, "length": len(w2_resp["text"]), "latency_ms": w2_resp["latency_ms"], "error": w2_resp["error"]}
                    )
                    execution_trace.record_step(
                        "Worker3_Parallel_PC3",
                        "API_ERROR" if w3_resp["error"] else "SUCCESS",
                        {"iteration": iteration, "length": len(w3_resp["text"]), "latency_ms": w3_resp["latency_ms"], "error": w3_resp["error"]}
                    )

                    node_latencies = {
                        "Worker1 (PC1 - RTX 4060)": w1_resp["latency_ms"],
                        "Worker2 (PC2 - Jetson Xavier)": w2_resp["latency_ms"],
                        "Worker3 (PC3 - MacBook M1 Pro)": w3_resp["latency_ms"]
                    }
                    straggler_node = max(node_latencies, key=node_latencies.get)
                    sum_latencies = sum(node_latencies.values())

                    execution_trace.record_step(
                        "Parallel_Stage_Summary",
                        "COMPLETED",
                        {
                            "wall_clock_ms": parallel_wall_time_ms,
                            "straggler_node": straggler_node,
                            "node_latencies_ms": node_latencies,
                            "time_saved_ms": round(sum_latencies - parallel_wall_time_ms, 2)
                        }
                    )

                    # Failure Injection Phase (applied to Worker 1's output before Judge sees it)
                    base_generation, is_poisoned = inject_hallucination_or_vulnerability(base_generation, request.mode)
                    if is_poisoned:
                        execution_trace.record_step("Failure_Injector", "POISONED", {"iteration": iteration, "target": "Worker1", "payload_type": request.mode})

                else:
                    # Sequential Pipeline: Worker 1 generates -> Failure Injector -> Work2 & Worker 3 audit concurrently
                    w1_resp = await query_manager(PC1_MANAGER_URL, "worker1", prompt=context_prompt, system_prompt=worker1_sys, temperature=0.7)
                    task_context.add_token_usage(w1_resp["prompt_tokens"], w1_resp["completion_tokens"])
                    base_generation = w1_resp["text"]

                    execution_trace.record_step(
                        "Worker1_Sequential_PC1",
                        "API_ERROR" if w1_resp["error"] else "SUCCESS",
                        {"iteration": iteration, "length": len(base_generation), "latency_ms": w1_resp["latency_ms"], "error": w1_resp["error"]}
                    )

                    base_generation, is_poisoned = inject_hallucination_or_vulnerability(base_generation, request.mode)
                    if is_poisoned:
                        execution_trace.record_step("Failure_Injector", "POISONED", {"iteration": iteration, "target": "Worker1", "payload_type": request.mode})

                    critique_prompt = f"User Request: {request.message}\n\nInitial Output from Worker 1:\n{base_generation}\n\nYour Audit:"
                    w2_resp, w3_resp = await asyncio.gather(
                        query_manager(PC2_MANAGER_URL, "worker2", prompt=critique_prompt, system_prompt=worker2_review_sys, temperature=0.2),
                        query_manager(PC3_MANAGER_URL, "worker3", prompt=critique_prompt, system_prompt=worker3_review_sys, temperature=0.3)
                    )
                    task_context.add_token_usage(w2_resp["prompt_tokens"], w2_resp["completion_tokens"])
                    task_context.add_token_usage(w3_resp["prompt_tokens"], w3_resp["completion_tokens"])

                    critique = (
                        f"[Worker 2 - SecOps Audit (PC2 Jetson)]:\n{w2_resp['text']}\n\n"
                        f"[Worker 3 - QA & Edge-Case Audit (PC3 M1 Pro)]:\n{w3_resp['text']}"
                    )

                    execution_trace.record_step(
                        "Worker2_Sequential_PC2",
                        "API_ERROR" if w2_resp["error"] else "SUCCESS",
                        {"iteration": iteration, "length": len(w2_resp["text"]), "latency_ms": w2_resp["latency_ms"], "error": w2_resp["error"]}
                    )
                    execution_trace.record_step(
                        "Worker3_Sequential_PC3",
                        "API_ERROR" if w3_resp["error"] else "SUCCESS",
                        {"iteration": iteration, "length": len(w3_resp["text"]), "latency_ms": w3_resp["latency_ms"], "error": w3_resp["error"]}
                    )

            else:
                # --- WORKER SELF-CORRECTION RETRY (iteration > 0) ---
                execution_trace.record_step("Feedback_Loop", "RETRY_TRIGGERED", {
                    "iteration": iteration,
                    "remediation_feedback": remediation_feedback
                })

                retry_prompt = (
                    f"{context_prompt}\n\n"
                    f"[PREVIOUS ATTEMPT REJECTED BY JUDGE]\n"
                    f"Rejected Output:\n{base_generation}\n\n"
                    f"Mandatory Remediation Instructions from Judge:\n{remediation_feedback}\n\n"
                    f"Provide a corrected, safe, and complete version fixing all reported issues:"
                )

                w1_resp = await query_manager(PC1_MANAGER_URL, "worker1", prompt=retry_prompt, system_prompt=worker1_sys, temperature=0.4)
                task_context.add_token_usage(w1_resp["prompt_tokens"], w1_resp["completion_tokens"])
                base_generation = w1_resp["text"]

                execution_trace.record_step(
                    "Worker1_Retry_PC1",
                    "API_ERROR" if w1_resp["error"] else "SUCCESS",
                    {"iteration": iteration, "length": len(base_generation), "latency_ms": w1_resp["latency_ms"], "error": w1_resp["error"]}
                )

                # Worker 2 and Worker 3 audit the newly corrected output concurrently
                critique_prompt = (
                    f"User Request: {request.message}\n\n"
                    f"Previous Judge Feedback: {remediation_feedback}\n\n"
                    f"Worker 1 Corrected Output:\n{base_generation}\n\n"
                    f"Verify if the flaws were properly fixed:"
                )
                w2_resp, w3_resp = await asyncio.gather(
                    query_manager(PC2_MANAGER_URL, "worker2", prompt=critique_prompt, system_prompt=worker2_review_sys, temperature=0.2),
                    query_manager(PC3_MANAGER_URL, "worker3", prompt=critique_prompt, system_prompt=worker3_review_sys, temperature=0.3)
                )
                task_context.add_token_usage(w2_resp["prompt_tokens"], w2_resp["completion_tokens"])
                task_context.add_token_usage(w3_resp["prompt_tokens"], w3_resp["completion_tokens"])

                critique = (
                    f"[Worker 2 - SecOps Audit (PC2 Jetson)]:\n{w2_resp['text']}\n\n"
                    f"[Worker 3 - QA & Edge-Case Audit (PC3 M1 Pro)]:\n{w3_resp['text']}"
                )

                execution_trace.record_step(
                    "Worker2_Retry_Audit_PC2",
                    "API_ERROR" if w2_resp["error"] else "SUCCESS",
                    {"iteration": iteration, "length": len(w2_resp["text"]), "latency_ms": w2_resp["latency_ms"], "error": w2_resp["error"]}
                )
                execution_trace.record_step(
                    "Worker3_Retry_Audit_PC3",
                    "API_ERROR" if w3_resp["error"] else "SUCCESS",
                    {"iteration": iteration, "length": len(w3_resp["text"]), "latency_ms": w3_resp["latency_ms"], "error": w3_resp["error"]}
                )

            # --- PHASE 2: JUDGE EVALUATION & SYNTHESIS (PC4) ---
            active_schema = (
                JudgeCodingVerdict.model_json_schema()
                if request.mode == "coding"
                else JudgeProVerdict.model_json_schema()
            )

            takeover_directive = (
                "CRITICAL: This is the final allowed iteration. Even if you set verdict to REJECT_*, "
                "you MUST personally write the complete, fixed, and sanitized solution in 'final_code'/'final_response' (Judge Takeover)."
                if is_last_attempt else
                "If you set verdict to REJECT_*, provide clear 'remediation_feedback' for Worker 1 to fix it in the next iteration, "
                "and set 'final_code'/'final_response' to 'PENDING_WORKER_RETRY' to save tokens."
            )

            judge_sys = (
                "You are the Lead Judge and Security Synthesizer. "
                "Evaluate Worker 1's output against Worker 2's SecOps audit and Worker 3's QA audit. "
                "Strictly flag any SQL Injection, command injection, missing validation, logical bugs, or hallucinations. "
                f"{takeover_directive}"
            )

            judge_user_prompt = (
                f"Iteration: {iteration} / {max_allowed_retries}\n"
                f"User Request: {request.message}\n\n"
                f"--- Worker 1 Output (PC1) ---\n{base_generation}\n\n"
                f"--- Council Audits (PC2 & PC3) ---\n{critique}\n"
            )

            judge_resp = await query_manager(
                PC4_MANAGER_URL,
                "judge",
                prompt=judge_user_prompt,
                system_prompt=judge_sys,
                temperature=0.0,
                json_schema=active_schema
            )
            task_context.add_token_usage(judge_resp["prompt_tokens"], judge_resp["completion_tokens"])
            raw_judge_output = judge_resp["text"]

            execution_trace.record_step(
                "Judge_Evaluation_PC4",
                "API_ERROR" if judge_resp["error"] else "SUCCESS",
                {"iteration": iteration, "latency_ms": judge_resp["latency_ms"], "finish_reason": judge_resp.get("finish_reason")}
            )

            # Validate GBNF JSON response via Pydantic
            try:
                cleaned_json = raw_judge_output.strip().removeprefix("```json").removesuffix("```").strip()
                if request.mode == "coding":
                    verdict_obj = JudgeCodingVerdict.model_validate_json(cleaned_json)
                    candidate_output = verdict_obj.final_code
                else:
                    verdict_obj = JudgeProVerdict.model_validate_json(cleaned_json)
                    candidate_output = verdict_obj.final_response

                parsed_verdict_dict = verdict_obj.model_dump()
                verdict_status = verdict_obj.verdict

                execution_trace.record_step("Judge_Validation", "SCHEMA_VALID", {
                    "iteration": iteration,
                    "verdict": verdict_status,
                    "security_score": parsed_verdict_dict.get("security_score", parsed_verdict_dict.get("factual_consistency_score")),
                    "flags_count": len(parsed_verdict_dict.get("security_flags", parsed_verdict_dict.get("detected_flaws", [])))
                })

            except Exception as e:
                print(f"Judge Pydantic validation failed on iteration {iteration}: {e}")
                execution_trace.record_step("Judge_Validation", "SCHEMA_FAILED", {"iteration": iteration, "error": str(e)})
                final_answer = raw_judge_output
                resolution_status = "SCHEMA_FALLBACK"
                break

            # --- PHASE 3: DECISION ROUTING (Approve vs Retry vs Takeover) ---
            if verdict_status == "APPROVE":
                final_answer = candidate_output
                resolution_status = "APPROVED_FIRST_PASS" if iteration == 0 else "APPROVED_AFTER_WORKER_RETRY"
                execution_trace.record_step("Pipeline_Resolution", resolution_status, {"iterations_total": iterations_used})
                break

            else:
                remediation_feedback = verdict_obj.remediation_feedback or "Fix all security and logic flaws identified in the audit."

                if not is_last_attempt:
                    print(f"[Iteration {iteration}] Judge issued {verdict_status}. Routing back to Worker 1...")
                    continue
                else:
                    print(f"[Iteration {iteration}] Max retries reached with {verdict_status}. Executing Judge Takeover...")
                    final_answer = candidate_output
                    resolution_status = f"JUDGE_TAKEOVER_AFTER_{verdict_status}"
                    execution_trace.record_step("Pipeline_Resolution", "JUDGE_TAKEOVER", {
                        "trigger_verdict": verdict_status,
                        "iterations_total": iterations_used
                    })
                    break

        update_history_with_response(request.session_id, final_answer)

        return {
            "final_response": final_answer,
            "resolution_status": resolution_status,
            "iterations": iterations_used,
            "judge_rubric": parsed_verdict_dict,
            "token_cost": task_context.get_stats_dict(),
            "execution_trace": execution_trace.trace,
            "debug": {
                "last_worker1_generation": base_generation,
                "last_council_critique": critique
            }
        }

    else:
        raise HTTPException(status_code=400, detail="Invalid mode selected.")
