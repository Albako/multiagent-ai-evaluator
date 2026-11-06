# MultiAgent AI Evaluator

## Hardware
My project is hosted on 5 different devices:

| Node | CPU | RAM | GPU | VRAM |
|:---:|:---:|:---:|:---:|:---:|
| **PC0**<br>Serves as an API<br>OS: Debian 13<br>Kernel: 7.0 | Intel Core i5-13500TE | 32GB ~35GB/s | No dedicated GPU | - |
| **PC1**<br>MSI Alpha 17 C7VF<br>OS: Fedora 44<br>Kernel: 7.2 | AMD Ryzen 9 7945HX<br>5.4 GHz<br>8 out of 16 cores | 64GB ~38GB/s | Nvidia RTX 4060 Mobile<br>3072 CUDA cores<br>96 Tensor cores | 8GB ~230GB/s |
| **PC2**<br>Nvidia Jetson AGX Xavier<br>Developer Kit<br>OS: Ubuntu 20 L4T<br>Kernel: 5.10-tegra | ARMv8.2 Carmel<br>2.27GHz<br>8 cores<br>MAXN mode enabled | 16GB ~65.5GB/s | iGPU<br>512 CUDA cores<br>64 Tensor cores | Unified memory<br>~114GB/s |
| **PC3**<br>MacBook Pro 16<br>OS: macOS 27.2 | Apple Silicon M1 Pro<br>3.2 GHz<br>8 P-cores | 16GB ~139GB/s | iGPU<br>2048 ALU | Unified memory<br>~182GB/s |
| **PC4**<br>Workstation<br>OS: Fedora 44<br>Kernel: 7.2 | Intel Core i7-14700KF<br>5.6GHz<br>8 P-cores | 64GB ~87.9GB/s | Nvidia RTX 4070 Ti<br>7680 CUDA cores<br>240 Tensor cores | 12GB ~440GB/s |

`PC1`, `PC3`, and `PC4` are pinned to 8 physical CPU cores (`P-cores` on Intel/Apple Silicon, single CCD on Ryzen) for optimal cache locality and memory bandwidth. Hardware parameters (`OVERRIDE_GPU_LAYERS`, `OVERRIDE_N_CTX`, `N_BATCH`) are managed per node in `docker-compose.yml` (and `start_mac.sh` on `PC3`).

---

## Architecture: Nodes, Roles and Model Families

The system separates **execution modes** (`fast`, `pro`, `coding`) from **model families** (`gemma`, `qwen`, `mixed`), allowing any model family to be benchmarked in any mode.

| Node | Role in Pipeline | `gemma` Family | `qwen` Family | `mixed` Family |
|:---:|:---|:---|:---|:---|
| **PC1** (`worker1`) | **Lead SecOps Generator** (Primary code/response generation) | `gemma-4-12B-it-qat-UD-Q4_K_XL` | `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q8_0` | `gemma-4-12B-it-qat-UD-Q4_K_XL` |
| **PC2** (`worker2`) | **Performance & Optimization Auditor** (Complexity & speed critique) | `gemma-4-12B-it-qat-UD-Q4_K_XL` | `Qwen3.5-9B-UD-Q6_K_XL` | `Qwen3.5-9B-UD-Q6_K_XL` |
| **PC3** (`worker3`) | **QA, Logic & Edge-Case Auditor** (Boundary conditions & CWE verification) | `gemma-4-12B-it-qat-UD-Q4_K_XL` | `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M` | `gemma-4-12B-it-qat-UD-Q4_K_XL` |
| **PC4** (`judge`) | **Lead Judge & Synthesizer** (Final evaluation, GBNF rubric, Takeover, or single-agent in `fast` mode) | `gemma-4-31B-it-qat-UD-Q4_K_XL` | `Qwen3.8-27B-UD-Q5_K_XL` | `Qwen3.8-27B-UD-Q5_K_XL` |

---

## Back-end: Starting the Cluster
Make sure to edit the IP addresses in the `.env` file (`.env` is automatically generated from `.env.example` on the first run of `./start.sh` or `./start_mac.sh`).

Start the services on each respective machine:

* **PC0 (Central API Gateway):**
  ```bash
  ./start.sh pc0
  ```
* **PC1 (Worker 1 - SecOps Generator):**
  ```bash
  ./start.sh pc1
  ```
* **PC2 (Worker 2 - Performance Auditor):**
  ```bash
  ./start.sh pc2
  ```
* **PC3 (Worker 3 - QA Auditor):**
  (Uses a native script instead of Docker to access Apple Silicon Metal GPU acceleration)
  ```bash
  ./start_mac.sh
  ```
* **PC4 (Judge Node):**
  ```bash
  ./start.sh pc4
  ```

## Testing Individual Nodes
Verify if a `<llm-model>.gguf` file loads properly into RAM/VRAM on individual nodes before initializing the whole cluster:
* **PC1:**
  ```bash
  docker exec -it model_manager_pc1 python3 -c "
  from llama_cpp import Llama
  print('--- Testing ---')
  llm = Llama(model_path='/models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf', n_ctx=1024, verbose=True)
  print('--- Success ---')
  "
  ```
* **PC2:**
  ```bash
  docker exec -it model_manager_pc2 python3 -c "
  from llama_cpp import Llama
  print('--- Testing ---')
  llm = Llama(model_path='/models/Qwen3.5-9B-UD-Q6_K_XL.gguf', n_gpu_layers=-1, n_ctx=1024, use_mmap=False, verbose=True)
  print('--- Success ---')
  "
  ```
* **PC3:**
  ```bash
  ./.venv_mac/bin/python3 -c "
  from llama_cpp import Llama
  print('--- Testing ---')
  llm = Llama(model_path='./models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf', n_gpu_layers=-1, n_ctx=4096)
  print('--- Success ---')
  "
  ```
* **PC4:**
  ```bash
  docker exec -it model_manager_pc4 python3 -c "
  from llama_cpp import Llama
  print('--- Testing ---')
  llm = Llama(model_path='/models/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf', n_ctx=1024, verbose=True)
  print('--- Success ---')
  "
  ```

## Initializing Models (`/system/init_mode`)
You can initialize any mode (`fast`, `pro`, `coding`) either with its default family defined in `.env` (`FAST_MODEL_FAMILY`, `PRO_MODEL_FAMILY`, `CODING_MODEL_FAMILY`) or by explicitly passing `"model_family"` (`"gemma"`, `"qwen"`, `"mixed"`).
If a node already has the target model loaded in VRAM with matching parameters, the manager skips reloading automatically.
1. *Initialize using default family from `.env`*
  * Coding:
   ```bash
   curl -s -X POST "[http://127.0.0.1:8000/system/init_mode](http://127.0.0.1:8000/system/init_mode)" \
        -H "Content-Type: application/json" \
        -d '{
              "mode": "coding"
            }' \
        -w "\n"
   ```
  * Pro:
    ```bash
    curl -s -X POST "[http://127.0.0.1:8000/system/init_mode](http://127.0.0.1:8000/system/init_mode)" \
         -H "Content-Type: application/json" \
         -d '{
               "mode": "pro"
             }' \
         -w "\n"
    ```
3. *Initialize specific model families (`gemma`, `qwen`, `mixed`)*
   * Gemma with `coding`:
     ```bash
     curl -s -X POST "[http://127.0.0.1:8000/system/init_mode](http://127.0.0.1:8000/system/init_mode)" \
          -H "Content-Type: application/json" \
          -d '{
                "mode": "coding",
                "model_family": "gemma"
              }' \
          -w "\n"
     ```
   * Gemma with `pro`:
     ```bash
     curl -s -X POST "[http://127.0.0.1:8000/system/init_mode](http://127.0.0.1:8000/system/init_mode)" \
          -H "Content-Type: application/json" \
          -d '{
                "mode": "pro",
                "model_family": "gemma"
              }' \
          -w "\n"
    * Gemma with `fast`:
      ```bash
      curl -s -X POST "[http://127.0.0.1:8000/system/init_mode](http://127.0.0.1:8000/system/init_mode)" \
           -H "Content-Type: application/json" \
           -d '{
                 "mode": "fast",
                 "model_family": "gemma"
               }' \
           -w "\n"
      ```
   * Qwen/Qwythos with `coding`:
      ```bash
      curl -s -X POST "[http://127.0.0.1:8000/system/init_mode](http://127.0.0.1:8000/system/init_mode)" \
           -H "Content-Type: application/json" \
           -d '{
                 "mode": "coding",
                 "model_family": "qwen"
               }' \
          -w "\n"
      ```
    * Qwen/Qwythos with `pro`:
      ```bash
      curl -s -X POST "[http://127.0.0.1:8000/system/init_mode](http://127.0.0.1:8000/system/init_mode)" \
          -H "Content-Type: application/json" \
          -d '{
                "mode": "pro",
                "model_family": "qwen"
              }' \
          -w "\n"
      ```
   * Qwen with `fast`:
      ```bash
      curl -s -X POST "[http://127.0.0.1:8000/system/init_mode](http://127.0.0.1:8000/system/init_mode)" \
           -H "Content-Type: application/json" \
           -d '{
                 "mode": "fast",
                 "model_family": "qwen"
               }' \
           -w "\n"
      ```
   * Mixed with `coding`:
     ```bash
     curl -s -X POST "[http://127.0.0.1:8000/system/init_mode](http://127.0.0.1:8000/system/init_mode)" \
          -H "Content-Type: application/json" \
          -d '{
                "mode": "coding",
                "model_family": "mixed"
              }' \
          -w "\n"
     ```
   * Mixed with `pro`:
     ```bash
     curl -s -X POST "[http://127.0.0.1:8000/system/init_mode](http://127.0.0.1:8000/system/init_mode)" \
          -H "Content-Type: application/json" \
          -d '{
                "mode": "pro",
                "model_family": "mixed"
              }' \
          -w "\n"
     ```
   * Force VRAM Reload (`"force_reload": true` flag):
     ```bash
     curl -s -X POST "[http://127.0.0.1:8000/system/init_mode](http://127.0.0.1:8000/system/init_mode)" \
          -H "Content-Type: application/json" \
          -d '{
                "mode": "coding",
                "model_family": "mixed",
                "force_reload": true
              }' \
          -w "\n"
     ```

## Running Inference (/chat)
Once the desired mode and model family are loaded, send requests to the `/chat` endpoint.
1. *Fast Mode (Solo PC4)*
   ```bash
   curl -X POST "[http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)" \
        -H "Content-Type: application/json" \
        -d '{
              "message": "Explain the difference between symmetric and asymmetric encryption.",
              "mode": "fast",
              "clear_session_after": true
            }' \
        -w "\n"
   ```
2. *Coding Mode - Parallel Pipeline (`worker1 || worker2 || worker3 -> Judge`)*
   All 3 workers execute concurrently, followed by the Judge synthesis and security evaluation:
   ```bash
   curl -X POST "[http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)" \
        -H "Content-Type: application/json" \
        -d '{
              "message": "Write a Python function to check if a number is prime and optimize its time complexity.",
              "mode": "coding",
              "pipeline_type": "parallel",
              "max_retries": 1,
              "clear_session_after": true
            }' \
        -w "\n"
   ```
3. *Coding Mode - Sequential Pipeline (`worker1 -> worker2 || worker3 audit -> Judge`)*
   Worker 1 generates the initial SecOps code first, then Worker 2 and Worker 3 audit Worker 1's actual output concurrently before passing it to the Judge:
   ```bash
   curl -X POST "[http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)" \
        -H "Content-Type: application/json" \
        -d '{
              "message": "Write a secure Python function that queries an SQLite database for a user profile by username.",
              "mode": "coding",
              "pipeline_type": "sequential",
              "max_retries": 1,
              "clear_session_after": true
            }' \
        -w "\n"
   ```
4. *Pro Mode (Factual Verification & Anti-Hallucination)*
   ```bash
   curl -X POST "[http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)" \
        -H "Content-Type: application/json" \
        -d '{
              "message": "Describe secure practices for deploying containerized services on Linux with rootless privileges.",
              "mode": "pro",
              "pipeline_type": "sequential",
              "max_retries": 1,
              "clear_session_after": true
            }' \
        -w "\n"
   ```
5. *Testing failure injection and self-correction loop*
   Force the failure injection (`"force_failure_injection": true`) to poison Worker 1's output with a CWE vulnerability / Indirect Prompt Injection and test wheter the Council and Judge reject and remediate it:
   ```bash
   curl -X POST "[http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)" \
        -H "Content-Type: application/json" \
        -d '{
              "message": "Write a Python function to authenticate a user from a database.",
              "mode": "coding",
              "pipeline_type": "sequential",
              "max_retries": 1,
              "force_failure_injection": true,
              "clear_session_after": true
            }' \
        -w "\n"
   ```
## Node Health and Memory Management
Check active models, KC-cashe quantization, GPU layers overrides, and peak RSS memory on any node (8001 - 8004):
  * **PC1:**
    ```bash
    curl -s "[http://127.0.0.1:8001/api/v1/system/status](http://127.0.0.1:8001/api/v1/system/status)" | python3 -m json.tool
    ```
  * **PC2:**
    ```bash
    curl -s "[http://127.0.0.1:8002/api/v1/system/status](http://127.0.0.1:8002/api/v1/system/status)" | python3 -m json.tool
    ```
  * **PC3:**
    ```bash
    curl -s "[http://127.0.0.1:8003/api/v1/system/status](http://127.0.0.1:8003/api/v1/system/status)" | python3 -m json.tool
    ```
  * **PC4:**
    ```bash
    curl -s "[http://127.0.0.1:8004/api/v1/system/status](http://127.0.0.1:8004/api/v1/system/status)" | python3 -m json.tool
    ```
Unload models from VRAM/RAM on a specific node:
  * **PC1:**
    ```bash
    curl -s -X POST "[http://127.0.0.1:8001/api/v1/system/clear_all](http://127.0.0.1:8001/api/v1/system/clear_all)" -w "\n"
    ```
  * **PC2:**
    ```bash
    curl -s -X POST "[http://127.0.0.1:8002/api/v1/system/clear_all](http://127.0.0.1:8002/api/v1/system/clear_all)" -w "\n"
    ```
  * **PC3:**
    ```bash
    curl -s -X POST "[http://127.0.0.1:8003/api/v1/system/clear_all](http://127.0.0.1:8003/api/v1/system/clear_all)" -w "\n"
    ```
  * **PC4:**
    ```bash
    curl -s -X POST "[http://127.0.0.1:8004/api/v1/system/clear_all](http://127.0.0.1:8004/api/v1/system/clear_all)" -w "\n"
    ```
