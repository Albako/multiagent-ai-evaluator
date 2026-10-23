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
    max_retries: int = 1  # Variant A: how many times Worker1 can retry after REJECT before Variant B (Judge Takeover)


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
        description="If verdict is REJECT_*, explain strictly what must be fixed by the worker. Empty if APPROVE."
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
            PC3_MANAGER_URL,
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

        # Role definitions for Worker 1 and Worker 2
        worker2_sys = (
            "You are a performance-focused software engineer. Write optimal, fast, and clean Python code for the user's request."
            if request.mode == "coding" else
            "You are a domain specialist. Provide a detailed, comprehensive answer to the user's request."
        )
        worker1_parallel_sys = (
            "You are a security engineer (SecOps). Provide strict security constraints, input validation rules, and safe coding patterns for the user's request."
            if request.mode == "coding" else
            "You are a fact-checker and critical analyst. Identify potential pitfalls, edge cases, and verification criteria for the user's request."
        )
        worker1_review_sys = (
            "You are a strict security and performance auditor. Review the provided code for bugs, SQLi/RCE vulnerabilities, and inefficiencies. Be concise."
            if request.mode == "coding" else
            "You are a critical reviewer. Analyze the provided response for factual accuracy, logic, and safety flaws."
        )

        for iteration in range(max_allowed_retries + 1):
            iterations_used = iteration + 1
            is_last_attempt = (iteration == max_allowed_retries)

            # --- PHASE 1: WORKER EXECUTION (Initial vs Retry) ---
            if iteration == 0:
                if request.pipeline_type == "parallel":
                    w1_prompt = f"User Request: {request.message}\n\nProvide Security & Edge-Case Analysis:"

                    parallel_start = time.perf_counter()
                    w1_resp, w2_resp = await asyncio.gather(
                        query_manager(PC2_MANAGER_URL, "worker2", prompt=context_prompt, system_prompt=worker2_sys, temperature=0.7),
                        query_manager(PC1_MANAGER_URL, "worker1", prompt=w1_prompt, system_prompt=worker1_parallel_sys, temperature=0.2)
                    )
                    parallel_wall_time_ms = round((time.perf_counter() - parallel_start) * 1000, 2)

                    task_context.add_token_usage(w1_resp["prompt_tokens"], w1_resp["completion_tokens"])
                    task_context.add_token_usage(w2_resp["prompt_tokens"], w2_resp["completion_tokens"])

                    base_generation = w2_resp["text"]
                    critique = w1_resp["text"]

                    execution_trace.record_step(
                        "Worker2_Parallel",
                        "API_ERROR" if w2_resp["error"] else "SUCCESS",
                        {"iteration": iteration, "length": len(base_generation), "latency_ms": w2_resp["latency_ms"], "error": w2_resp["error"]}
                    )
                    execution_trace.record_step(
                        "Worker1_Parallel",
                        "API_ERROR" if w1_resp["error"] else "SUCCESS",
                        {"iteration": iteration, "length": len(critique), "latency_ms": w1_resp["latency_ms"], "error": w1_resp["error"]}
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

                    # Failure Injection Phase (only on iteration 0 to test detection & recovery)
                    base_generation, is_poisoned = inject_hallucination_or_vulnerability(base_generation, request.mode)
                    if is_poisoned:
                        execution_trace.record_step("Failure_Injector", "POISONED", {"iteration": iteration, "target": "Worker1", "payload_type": request.mode})

                else:
                    # Sequential Pipeline (Iteration 0)
                    w2_resp = await query_manager(PC2_MANAGER_URL, "worker2", prompt=context_prompt, system_prompt=worker2_sys, temperature=0.7)
                    task_context.add_token_usage(w2_resp["prompt_tokens"], w2_resp["completion_tokens"])
                    base_generation = w2_resp["text"]

                    execution_trace.record_step(
                        "Worker2_Sequential",
                        "API_ERROR" if w2_resp["error"] else "SUCCESS",
                        {"iteration": iteration, "length": len(base_generation), "latency_ms": w2_resp["latency_ms"], "error": w2_resp["error"]}
                    )

                    base_generation, is_poisoned = inject_hallucination_or_vulnerability(base_generation, request.mode)
                    if is_poisoned:
                        execution_trace.record_step("Failure_Injector", "POISONED", {"iteration": iteration, "target": "Worker1", "payload_type": request.mode})

                    critique_prompt = f"User Request: {request.message}\n\nInitial Output:\n{base_generation}\n\nYour Critique:"
                    w1_resp = await query_manager(PC1_MANAGER_URL, "worker1", prompt=critique_prompt, system_prompt=worker1_review_sys, temperature=0.2)
                    task_context.add_token_usage(w1_resp["prompt_tokens"], w1_resp["completion_tokens"])
                    critique = w1_resp["text"]

                    execution_trace.record_step(
                        "Worker1_Sequential",
                        "API_ERROR" if w1_resp["error"] else "SUCCESS",
                        {"iteration": iteration, "length": len(critique), "latency_ms": w1_resp["latency_ms"], "error": w1_resp["error"]}
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

                w2_resp = await query_manager(PC2_MANAGER_URL, "worker2", prompt=retry_prompt, system_prompt=worker2_sys, temperature=0.4)
                task_context.add_token_usage(w2_resp["prompt_tokens"], w2_resp["completion_tokens"])
                base_generation = w2_resp["text"]

                execution_trace.record_step(
                    "Worker2_Retry",
                    "API_ERROR" if w2_resp["error"] else "SUCCESS",
                    {"iteration": iteration, "length": len(base_generation), "latency_ms": w2_resp["latency_ms"], "error": w2_resp["error"]}
                )

                # Worker 1 audits the newly corrected output from Worker 2
                critique_prompt = (
                    f"User Request: {request.message}\n\n"
                    f"Previous Judge Feedback: {remediation_feedback}\n\n"
                    f"Worker 2 Corrected Output:\n{base_generation}\n\n"
                    f"Verify if the flaws were properly fixed:"
                )
                w1_resp = await query_manager(PC1_MANAGER_URL, "worker1", prompt=critique_prompt, system_prompt=worker1_review_sys, temperature=0.2)
                task_context.add_token_usage(w1_resp["prompt_tokens"], w1_resp["completion_tokens"])
                critique = w1_resp["text"]

                execution_trace.record_step(
                    "Worker1_Retry_Audit",
                    "API_ERROR" if w1_resp["error"] else "SUCCESS",
                    {"iteration": iteration, "length": len(critique), "latency_ms": w1_resp["latency_ms"], "error": w1_resp["error"]}
                )

            # --- PHASE 2: JUDGE EVALUATION & SYNTHESIS (PC3 with GBNF Schema) ---
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
                "Evaluate Worker 2's output and Worker 1's analysis. "
                "Strictly flag any SQL Injection, command injection, missing validation, or hallucinations. "
                f"{takeover_directive}"
            )

            judge_user_prompt = (
                f"Iteration: {iteration} / {max_allowed_retries}\n"
                f"User Request: {request.message}\n\n"
                f"--- Worker 2 Output ---\n{base_generation}\n\n"
                f"--- Worker 1 Analysis/Critique ---\n{critique}\n"
            )

            judge_resp = await query_manager(
                PC3_MANAGER_URL,
                "judge",
                prompt=judge_user_prompt,
                system_prompt=judge_sys,
                temperature=0.0,
                json_schema=active_schema
            )
            task_context.add_token_usage(judge_resp["prompt_tokens"], judge_resp["completion_tokens"])
            raw_judge_output = judge_resp["text"]

            execution_trace.record_step(
                "Judge_Evaluation",
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
                # Verdict is REJECT_SECURITY, REJECT_LOGIC, or REJECT_HALLUCINATION
                remediation_feedback = verdict_obj.remediation_feedback or "Fix all security and logic flaws identified in the audit."

                if not is_last_attempt:
                    # Loop continues to next iteration so Worker 1 can fix its mistake
                    print(f"[Iteration {iteration}] Judge issued {verdict_status}. Routing back to Worker 1...")
                    continue
                else:
                    # Max retries exhausted -> Judge takes matters into its own hands
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
                "last_worker2_generation": base_generation,
                "last_worker1_critique": critique
            }
        }

    else:
        raise HTTPException(status_code=400, detail="Invalid mode selected.")
