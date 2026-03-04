import os
import random
import asyncio
import httpx
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MultiAgent Chat API Gateway")

chat_histories = {}

PC1_MANAGER_URL = os.getenv("PC1_MANAGER_URL", "http://localhost:8001")
PC2_MANAGER_URL = os.getenv("PC2_MANAGER_URL", "http://localhost:8003")

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
            raise HTTPException(status_code=500, detail="Missing FAST_MODEL in .env file")

        # fast mode only works on pc1 with worker1
        await load_model_on_manager(PC1_MANAGER_URL, "worker1", fast_model)
        return {"status": "success", "message": "Fast mode model loaded into VRAM"}

    elif mode == "pro":
        worker1_model = os.getenv("PRO_WORKER_1_MODEL")
        worker2_model = os.getenv("PRO_WORKER_2_MODEL")
        judge_model = os.getenv("PRO_JUDGE_MODEL")

        if not all([worker1_model, worker2_model, judge_model]):
            raise HTTPException(status_code=500, detail="Missing model configuration in .env file for Pro mode")

        await load_model_on_manager(PC1_MANAGER_URL, "worker1", worker1_model)
        await load_model_on_manager(PC1_MANAGER_URL, "worker2", worker2_model)
        await load_model_on_manager(PC2_MANAGER_URL, "judge", judge_model)
        return {"status": "success", "message": "Pro mode model loaded into VRAM."}

    elif mode == "coding":
        worker1_model = os.getenv("CODING_WORKER_1_MODEL")
        worker2_model = os.getenv("CODING_WORKER_2_MODEL")
        judge_model = os.getenv("CODING_JUDGE_MODEL")

        if not all([worker1_model, worker2_model, judge_model]):
            raise HTTPException(status_code=500, detail="Missing model configuration in .env file for Coding mode")

        await load_model_on_manager(PC1_MANAGER_URL, "worker1", worker1_model)
        await load_model_on_manager(PC1_MANAGER_URL, "worker2", worker2_model)
        await load_model_on_manager(PC2_MANAGER_URL, "judge", judge_model)
        return {"status": "success", "message": "Coding mode model loaded into VRAM."}

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

async def query_manager(url: str, worker_id: str, prompt: str) -> str:
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{url}/api/v1/{worker_id}/generate",
                json={"prompt": prompt, "max_tokens": 2048, "temperature": 0.7}
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

    elif request.mode in ["pro", "coding"]:
        max_iterations = 3
        iteration = 0
        approved = False

        current_prompt = context_prompt
        final_answer = ""

        while iteration < max_iterations and not approved:
            print(f"--- Starting Iteration {iteration + 1} ---")
            # pro/coding mode - both workers are loaded in the same time
            worker1_task = query_manager(PC1_MANAGER_URL, "worker1", request.message)
            worker2_task = query_manager(PC1_MANAGER_URL, "worker2", request.message)
            worker1_response, worker2_response = await asyncio.gather(worker1_task, worker2_task)

            # blind evaluation with order shuffling
            responses = [worker1_response, worker2_response]
            random.shuffle(responses)

            judge_prompt = (
                "You are a strict judge. Truth is the most important thing for you. You don't like responses that seem true, but aren't.\n\n"
                f"The user asked: '{request.message}'.\n\n"
                f"Response A: {responses[0]}\n\n"
                f"Response B: {responses[1]}\n\n"
                "Evaluate both responses critically. If at least one is correct, or if you can synthesize a perfect final answer from them, you MUST start your response exactly with 'STATUS: APPROVED', followed by 'FINAL_ANSWER: ' and then write the final correct response. If BOTH are flawed, incomplete, or incorrect, you MUST start your response exactly with 'STATUS: REJECTED', followed by 'CRITIQUE: ' and explain what is wrong and how the workers should fix it."
            )

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
