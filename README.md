# MultiAgent AI Evaluator

## Hardware
My project is hosted on 5 different devices:
1. PC0 API: My x86 server.
2. PC1 Worker1: R9 7945HX 64GB 83GB/s + RTX 4060 (mobile) 8GB 256GB/s
3. PC2 Worker2: Nvidia Jetson AGX Xavier Developer Kit 16GB 137GB/s with MAXN mode enabled
4. PC3 Worker3: MacBook Pro M1 Pro 16GB 200GB/s
5. PC4 Judge: i7-14700KF 64GB 102GB/s + RTX 4070 Ti 12GB 504GB/s

## Back-end
### Starting the cluster
Make sure to edit the IP adresses in the `.env` file (`.env` will create itself from `.env.example` after running the `./start.sh pcX` script).

In order to start the cluster use this command in the API host:
```bash
./start.sh pc0
```
and in the R9+4060 x86 (Worker1):
```bash
./start.sh pc1
```
and in the Nvidia Jeston AGX Xavier arm64 (Worker2):
```bash
./start.sh pc2
```
and in the MacBook arm64 (Worker3) - MacBook has a seperate script because it doesn't use Docker, because Docker can't use Apple Silicone GPU:
```bash
./start_mac.sh
```
and finally the i7+4070Ti x86 (Judge):
```bash
./start.sh pc4
```

## Testing
Testing if a `llm-model.gguf` can be loaded on individual nodes:
PC1:
```bash
docker exec -it model_manager_pc1 python3 -c "
from llama_cpp import Llama
print('--- Testing ---')
llm = Llama(model_path='/models/gemma-4-12b-it-UD-Q4_K_XL.gguf', n_ctx=1024, verbose=True)
print('--- Success ---')
"
```
PC2:
```bash
docker exec -it model_manager_pc2 python3 -c "
from llama_cpp import Llama
print('--- Testing ---')
llm = Llama(model_path='/models/gemma-4-E4B-it-Q6_K.gguf', n_gpu_layers=24, n_ctx=1024, use_mmap=False, verbose=True)
print('--- Success ---')
"
```
PC3:
```bash

```
PC4:
```bash
docker exec -it model_manager_pc4 python3 -c "
from llama_cpp import Llama
print('--- Testing ---')
llm = Llama(model_path='/models/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf', n_ctx=1024, verbose=True)
print('--- Success ---')
"
```
## Starting the project
Start by loading the models:
```bash
curl -s -X POST "http://127.0.0.1:8000/system/init_mode" \
     -H "Content-Type: application/json" \
     -d '{
           "mode": "coding"
         }' \
     -w "\n"
```
Then send a prompt:
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
           "message": "Write a Python function to check if a number is prime and optimize its time complexity.",
           "mode": "coding"
         }' \
     -w "\n"
```
