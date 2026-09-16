import os
import random
import asyncio
import httpx
import re
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

# Class to track token budget across the entire pipeline
class TaskContext:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def add_token_usage(self, prompt_tokens: int, completion_tokens: int):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

    def get_stats_dict(self):
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens
        }

# Class to track error propagation and step localization
class ExecutionTrace:
    def __init__(self):
        self.trace = []

    def record_step(self, component: str, status: str, details: dict = None):
        self.trace.append({
            "component": component,
            "status": status,
            "details": details or {}
        })

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
        except httpx.RequestError as e:
            print(f"Failed to load {model_name} on {worker_id} at {url}: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

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

        await load_model_on_manager(PC1_MANAGER_URL, "worker1", worker1_model, n_gpu_layers=5)
        await load_model_on_manager(PC2_MANAGER_URL, "worker2", worker2_model, n_gpu_layers=-1)
        await load_model_on_manager(PC3_MANAGER_URL, "judge", judge_model, n_gpu_layers=10)
        return {"status": "success", "message": f"{mode.capitalize()} mode distributed across 3 nodes."}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")

class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str
    mode: str = "fast" # fast mode

def estimate_tokens(text: str) -> int:
    # estimation: 1 token is 4 characters
    return len(text) // 4

def apply_sliding_window(session_id: str, new_message: str, max_tokens: int = 4000) -> str:
    global chat_histories

    if session_id not in chat_histories:
        chat_histories[session_id] = ""

    history = chat_histories[session_id]

    # append the new user message
    history += f"\nUser: {new_message}\nAI: "

    # trim the history from the beggining if it exceeds the max tokens
    while estimate_tokens(history) > max_tokens:
        # find the first occurance of "User: " after index 0 to cut safely
        cut_index = history.find("\nUser: ", 1)
        if cut_index == -1:
            # if no other user prompt is found, just keep the current one to avoid breaking
            break
        history = history[cut_index:]

    chat_histories[session_id] = history
    return history

def update_history_with_response(session_id: str, response: str):
    global chat_histories
    chat_histories[session_id] += f"{response}\n"

# Modified to return a dictionary with text, tokens, and errors
async def query_manager(url: str, worker_id: str, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> dict:
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{url}/api/v1/{worker_id}/generate",
                json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
            )
            if response.status_code != 200:
                print(f"Error from API: {response.text}")
            response.raise_for_status()
            data = response.json()
            return {
                "text": data.get("generated_text", ""),
                "prompt_tokens": data.get("prompt_tokens", 0),
                "completion_tokens": data.get("completion_tokens", 0),
                "error": None
            }
        except httpx.RequestError as e:
            print(f"HTTP Request failed: {e}")
            return {
                "text": f"Error communicating with {worker_id}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": str(e)
            }

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):

    context_prompt = apply_sliding_window(request.session_id, request.message)
    
    # Initialize analysis tracking tools
    task_context = TaskContext()
    execution_trace = ExecutionTrace()
    
    execution_trace.record_step("Gateway", "INITIALIZED", {"mode": request.mode})

    if request.mode == "fast":
        # fast mode - bypass workers, go straight to PC1
        resp_data = await query_manager(PC3_MANAGER_URL, "judge", context_prompt)
        task_context.add_token_usage(resp_data["prompt_tokens"], resp_data["completion_tokens"])
        response = resp_data["text"]
        
        if resp_data["error"]:
            execution_trace.record_step("Judge_Fast", "API_ERROR", {"error": resp_data["error"]})
        else:
            execution_trace.record_step("Judge_Fast", "SUCCESS", {"length": len(response)})

        update_history_with_response(request.session_id, response)
        return {
            "final_response": response, 
            "iterations": 0,
            "token_cost": task_context.get_stats_dict(),
            "execution_trace": execution_trace.trace
        }

    elif request.mode in ["pro", "coding"]:
        # Implements the Council Architecture: Generator -> Reviewer -> Unifier(Judge)
        print(f"--- Starting Council Pipeline for {request.mode} mode ---")

        # Step 1: Base Generation
        worker1_system_prompt = (
            "You are an expert engineer. Write the best possible response/code for the following request." 
            if request.mode == "coding" else 
            "You are an expert assistant. Provide a comprehensive response to the user's request."
        )
        w1_resp = await query_manager(PC1_MANAGER_URL, "worker1", f"{worker1_system_prompt}\n\n{context_prompt}", temperature=0.7)
        task_context.add_token_usage(w1_resp["prompt_tokens"], w1_resp["completion_tokens"])
        base_generation = w1_resp["text"]
        
        if w1_resp["error"]:
            execution_trace.record_step("Worker1", "API_ERROR", {"error": w1_resp["error"]})
        else:
            execution_trace.record_step("Worker1", "SUCCESS", {"length": len(base_generation)})
            
        # Failure Injection Phase
        base_generation, is_poisoned = inject_hallucination_or_vulnerability(base_generation, request.mode)
        if is_poisoned:
            execution_trace.record_step("Failure_Injector", "POISONED", {"payload_type": request.mode})

        # Step 2: Council Critique
        worker2_system_prompt = (
            "You are a strict security and performance auditor. Review the following code for bugs, vulnerabilities, and inefficiencies. Be practical and concise."
            if request.mode == "coding" else
            "You are a critical reviewer. Analyze the following response for factual accuracy, logic, and clarity. Point out any flaws."
        )
        critique_prompt = f"{worker2_system_prompt}\n\nUser Request: {request.message}\n\nInitial Output:\n{base_generation}\n\nYour Critique:"
        w2_resp = await query_manager(PC2_MANAGER_URL, "worker2", critique_prompt, temperature=0.2)
        task_context.add_token_usage(w2_resp["prompt_tokens"], w2_resp["completion_tokens"])
        critique = w2_resp["text"]
        
        if w2_resp["error"]:
            execution_trace.record_step("Worker2", "API_ERROR", {"error": w2_resp["error"]})
        else:
            execution_trace.record_step("Worker2", "SUCCESS", {"length": len(critique)})

        # Step 3: Judge Synthesis
        coding_instruction = (
            "You MUST respond ONLY with a valid JSON containing 'verdict', 'base_code', 'security_flags', 'performance_notes', 'summary'."
            if request.mode == "coding"
            else "Provide the final unified response, fixing the issues raised by the council."
        )

        judge_system_prompt = f"""You are the Lead Judge and Synthesizer.
        You are reviewing an initial response and feedback from an AI Council member.
        Your task is to produce the final, unified output incorporating valid feedback.

        User Request: {request.message}

        Initial Output (Worker 1):
        {base_generation}

        Council Critique (Worker 2):
        {critique}

        {coding_instruction}
        """
        
        judge_resp = await query_manager(PC3_MANAGER_URL, "judge", judge_system_prompt, temperature=0.0)
        task_context.add_token_usage(judge_resp["prompt_tokens"], judge_resp["completion_tokens"])
        final_answer = judge_resp["text"]
        
        if judge_resp["error"]:
            execution_trace.record_step("Judge", "API_ERROR", {"error": judge_resp["error"]})
        else:
            execution_trace.record_step("Judge", "SUCCESS", {"length": len(final_answer)})

        if request.mode == "coding":
            try:
                # Strip potential markdown code fences
                cleaned_json = final_answer.strip().removeprefix("```json").removesuffix("```").strip()
                json.loads(cleaned_json) # Test if it's valid JSON
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
