# MultiAgent AI Evaluator

## Hardware
My project is hosted on 4 different hosts:
1. PC0 API: My x86 server.
2. PC1 Worker1: R9 7945HX 64GB 5200MT/s + RTX 4060 (mobile) 8GB GDDR6
3. PC2 Worker2: Nvidia Jetson AGX Xavier Developer Kit 16GB 4267MT/s with MAXN mode enabled
4. PC3 Judge: i7-14700KF 64GB 6400MT/s + RTX 4070 Ti 12GB GDDR6X

## Back-end
### Starting the cluster
Make sure to edit the IP adresses in the `.env` file (`.env` will create itself from `.env.example` after running the `./start.sh pcx` script).

In order to start the cluster use this command in the API host:
```bash
./start.sh pc0
```
and in the R9+4060 PCx86 (Worker1):
```bash
./start.sh pc1
```
and in the Nvidia Jeston AGX Xavier PCarm64 (Worker2):
```bash
./start.sh pc2
```
and finally the i7+4070Ti PCx86 (Judge):
```bash
./start.sh pc3
```

## Testing
### Testing if an `.gguf` can be loaded on individual nodes:
PC2:
```bash
docker exec -it model_manager_pc2 python3 -c "
from llama_cpp import Llama
print('--- Testing ---')
llm = Llama(model_path='/models/gemma-4-12b-it-UD-Q4_K_XL.gguf', n_gpu_layers=24, n_ctx=1024, use_mmap=False, verbose=True)
print('--- Success ---')
"
```
Start by loading the models:
```bash
curl -s -X POST "http://127.0.0.1:8000/system/init_mode" \
     -H "Content-Type: application/json" \
     -d '{
           "mode": "coding"
         }' \
     -w "\n"
```
Then test the connection:
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
           "message": "Write a Python function to check if a number is prime and optimize its time complexity.",
           "mode": "coding"
         }' \
     -w "\n"
```
