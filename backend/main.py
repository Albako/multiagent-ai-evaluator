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

class InitModeRequest(BaseModel):
    mode: str

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
        await load_model_on_manager(PC1_MANAGER_URL, "worker1", fast_model)
        return {"status": "success", "message": "Fast mode loaded on PC1"}

    elif mode in ["pro", "coding"]:
        prefix = "PRO" if mode == "pro" else "CODING"
        worker1_model = os.getenv(f"{prefix}_WORKER_1_MODEL")
        worker2_model = os.getenv(f"{prefix}_WORKER_2_MODEL")
        judge_model = os.getenv(f"{prefix}_JUDGE_MODEL")

        if not all([worker1_model, worker2_model, judge_model]):
            raise HTTPException(status_code=500, detail=f"Missing models for {mode} mode")

        await load_model_on_manager(PC1_MANAGER_URL, "worker1", worker1_model)
        await load_model_on_manager(PC2_MANAGER_URL, "worker2", worker2_model)
        await load_model_on_manager(PC3_MANAGER_URL, "judge", judge_model)
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

async def query_manager(url: str, worker_id: str, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{url}/api/v1/{worker_id}/generate",
                json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
            )
            response.raise_for_status()
            data = response.json()
            return data["response"]["choices"][0]["text"].strip()
        except httpx.RequestError as e:
            print(f"HTTP Request failed: {e}")
            return f"Error communicating with {worker_id}"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):

    context_prompt = apply_sliding_window(request.session_id, request.message)

    if request.mode == "fast":
        # fast mode - bypass workers, go straight to PC1
        response = await query_manager(PC1_MANAGER_URL, "worker1", request.message)
        return {"final_response": response, "iterations": 0}

    elif request.mode in ["pro"]:
        max_iterations = 3
        iteration = 0
        approved = False

        current_prompt = context_prompt
        final_answer = ""

        while iteration < max_iterations and not approved:
            print(f"--- Starting Iteration {iteration + 1} ---")
            # pro/coding mode - both workers are loaded in the same time
            worker1_task = query_manager(PC1_MANAGER_URL, "worker1", current_prompt)
            worker2_task = query_manager(PC1_MANAGER_URL, "worker2", current_prompt)
            worker1_response, worker2_response = await asyncio.gather(worker1_task, worker2_task)

            # blind evaluation with order shuffling
            responses = [worker1_response, worker2_response]
            random.shuffle(responses)

            judge_pro_prompt = f"""You are an objective evaluator focusing on factual accuracy, reasoning, and style.
            Evaluate two responses to the user request: '{request.message}'.

            Response A: {responses[0]}
            Response B: {responses[1]}

            Guidelines:
            - Do not combine directly contradictory facts. Choose the one that is factually correct.
            - Evaluate in a binary way: Is the response helpful and accurate (Yes/No)?
            - If both are highly inaccurate, reject them.

            Examples of good synthesis:
            User: Explain black holes and their temperature.
            A: Black holes emit Hawking radiation.
            B: Black holes are extremely cold, but accretion disks are hot.
            Synthesis: Choose both facts, as they do not contradict and provide a full picture.

            Examples of bad synthesis:
            User: Who won the 2002 World Cup?
            A: Brazil won in 2002.
            B: Germany won in 2002.
            Synthesis: DO NOT combine. Brazil is correct. Germany is wrong.

            Provide your final response containing:
            1. THE FINAL ANSWER (clearly separated).
            2. A REPORT section containing:
            - Which model did better in merit vs style.
            - Any conflicting facts detected and which was chosen and why.
            - Justification for your synthesis approach.

            Start your response with 'STATUS: APPROVED' or 'STATUS: REJECTED'.
            """

            judge_response = await query_manager(PC3_MANAGER_URL, "judge", judge_pro_prompt, temperature=0.7)

            # asking the judge
            judge_response = await query_manager(PC2_MANAGER_URL, "judge", judge_prompt)
            judge_response_upper = judge_response.upper()

            if "STATUS: APPROVED" in judge_response_upper:
                approved = True
                # extracting the final answer
                if "FINAL_ANSWER:" in judge_response_upper:
                    # find original casing index
                    idx = judge_response_upper.find("FINAL_ANSWER:") + len("FINAL_ANSWER:")
                    critique = judge_response[idx:].strip()
                else:
                    # fallback if judge has dementia and forgets the exact tag
                    final_answer = re.sub(r'(?i)STATUS:\s*APPROVED', '', judge_response).strip()

            elif "STATUS: REJECTED" in judge_response_upper:
                if "CRITIQUE:" in judge_response_upper:
                    idx = judge_response_upper.find("CRITIQUE:") + len("CRITIQUE:")
                    critique = judge_response[idx:].strip()
                else:
                    critique = re.sub(r'(?i)STATUS:\s*REJECTED', '', judge_response).strip()

                # update the prompt for the next iteration
                current_prompt = f"Original request: {request.message}\n\nJudge's critique on your previous attempt: {critique}\n\nPlease provide an improved response fixing these issues."
                iteration += 1

            else:
                # fallback if judge is hallucinating
                print(f"Judge failed to use strict formatting. Forcing approval to avoid crash.")
                approved = True
                final_answer = judge_response

    elif request.mode in ["coding"]:
        max_iterations = 3
        iteration = 0
        approved = False

        current_prompt = context_prompt
        final_answer = ""
        worker1_system_prompt = "You are an expert software engineer focusing strictly on runtime performance, algorithmic efficiency, and execution speed. Write the best possible high-performance code for the following request: "
        worker2_system_prompt = "You are an expert software security engineer. You focus entirely on code security, robust error handling, preventing vulnerabilities, and edge-case safety. Write the most secure code for the following request: "

        while iteration < max_iterations and not approved:
            print(f"--- Starting Iteration {iteration + 1} ---")
            # pro/coding mode - both workers are loaded in the same time
            worker1_task = query_manager(PC1_MANAGER_URL, "worker1", f"{worker1_system_prompt}\n{request.message}")
            worker2_task = query_manager(PC2_MANAGER_URL, "worker2", f"{worker2_system_prompt}\n{request.message}")
            worker1_response, worker2_response = await asyncio.gather(worker1_task, worker2_task)

            # blind evaluation with order shuffling
            responses = [worker1_response, worker2_response]
            random.shuffle(responses)


            judge_coder_prompt = f"""You are a strict, methodical code judge. You must evaluate two code snippets.
            Response A: {responses[0]}
            Response B: {responses[1]}

            Follow these steps exactly (Chain of Thought):
            1. Analyze the computational complexity and performance of both codes.
            2. Analyze the security, error handling, and bounds checking of both codes.
            3. Check for syntax errors, proper indentation, and missing imports.
            4. If both codes are fundamentally broken and do not compile, declare a tie ("REJECTED_BOTH"). Do not force a choice.
            5. If at least one is viable, select one as the foundation. Synthesize a final code by rewriting the foundation from scratch, injecting the missing optimizations or security features from the other code.

            You MUST respond ONLY in the following JSON format. No markdown blocks, no extra text:
            {{
                "status": "APPROVED" or "REJECTED_BOTH" or "REJECTED",
                "critique": "Explanation if rejected, or empty if approved.",
                "report": {{
                    "foundation_chosen": "Response A, Response B, or Tie",
                    "performance_security_improvements": "Details on what was merged...",
                    "syntax_and_import_check": "Confirmation of syntax and imports..."
                }},
                "reasoning": "Your step-by-step chain of thought analysis...",
                "final_code": "The synthesized, fully working code."
            }}
            """

            # Fetching from Judge with low temperature
            judge_response = await query_manager(PC3_MANAGER_URL, "judge", judge_coder_prompt, temperature=0.0)

            # asking the judge
            judge_response = await query_manager(PC2_MANAGER_URL, "judge", judge_prompt)
            judge_response_upper = judge_response.upper()

            if "STATUS: APPROVED" in judge_response_upper:
                approved = True
                # extracting the final answer
                if "FINAL_ANSWER:" in judge_response_upper:
                    # find original casing index
                    idx = judge_response_upper.find("FINAL_ANSWER:") + len("FINAL_ANSWER:")
                    critique = judge_response[idx:].strip()
                else:
                    # fallback if judge has dementia and forgets the exact tag
                    final_answer = re.sub(r'(?i)STATUS:\s*APPROVED', '', judge_response).strip()

            elif "STATUS: REJECTED" in judge_response_upper:
                if "CRITIQUE:" in judge_response_upper:
                    idx = judge_response_upper.find("CRITIQUE:") + len("CRITIQUE:")
                    critique = judge_response[idx:].strip()
                else:
                    critique = re.sub(r'(?i)STATUS:\s*REJECTED', '', judge_response).strip()

                # update the prompt for the next iteration
                current_prompt = f"Original request: {request.message}\n\nJudge's critique on your previous attempt: {critique}\n\nPlease provide an improved response fixing these issues."
                iteration += 1

            else:
                # fallback if judge is hallucinating
                print(f"Judge failed to use strict formatting. Forcing approval to avoid crash.")
                approved = True
                final_answer = judge_response

        # if iterations == 3 and still answers are awful, then force a final resolution
        if not approved:
            print("Max iterations reached. Forcing final synthesis.")
            force_prompt = f"Synthesize the best possible answer from these two flawed attempts based on the user request: '{request.message}'.\nA: {worker1_response}\nB: {worker2_response}\nOutput ONLY the final answer."
            final_answer = await query_manager(PC2_MANAGER_URL, "judge", force_prompt)

        update_history_with_response(request.session_id, final_answer)

        return {
            "final_response": final_answer,
            "iterations": iteration + 1 if not approved else iteration,
            "debug": {
                "worker1_last": worker1_response,
                "worker2_last": worker2_response,
                "judge_raw": judge_response
            }
        }

    else:
        raise HTTPException(status_code=400, detail="Invalid mode selected.")
