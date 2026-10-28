# MultiAgent AI Evaluator

## Hardware
My project is hosted on 5 different devices:
|               Node                                         |                            CPU                            |      RAM     |                           GPU                                 |                               VRAM                          |
|:----------------------------------------------------------:|:---------------------------------------------------------:|:------------:|:-------------------------------------------------------------:|:-----------------------------------------------------------:|
|   PC0<br> Serves as an API<br> Debian 13<br> Kernel: 7.0   |                  Intel Core i5-13500TE<br>                |  32GB 43GB/s |           It yet doesn't have a dedicated GPU                 |                               -                             |
|   PC1<br> MSI Alpha 17 C7VF<br> Fedora 44<br> Kernel: 7.2  |  AMD Ryzen 9 7945HX<br> 5.4 GHz<br> 8 out of 16 cores     |  64GB 83GB/s |Nvidia RTX 4060 Mobile<br> 3072 CUDA cores<br> 96 Tensor cores |                            8GB 256GB/s                      |
|PC2<br> Nvidia Jetson AGX Xavier<br> Developer Kit<br> Ubuntu 20<br> Kernel: 5.10-tegra|ARMv8.2 Carmel<br> 2.27GHz<br> 8 cores<br> MAXN mode enabled | 16GB 137GB/s |iGPU<br> 512 CUDA cores<br> 64 Tensor cores|Unified memory<br> `USE_MMAP=false`<br> because of an old kernel|
|                  PC3<br> MacBook Pro 16                | Apple Silicone M1 Pro<br> 3.2 GHz<br> 8 Performance cores     | 16GB 200GB/s |                    iGPU<br> 1792 ALUs                     |                        Unified memory                       |
|  PC4<br> Workstation<br> Fedora 44<br> Kernel: 7.2     |    Intel Core i7-14700KF<br> 5.6GHz<br> 8 Performance cores   | 64GB 102GB/s |Nvidia RTX 4070 Ti<br> 7680 CUDA cores<br> 240 Tensor cores|                          12GB 504GB/s                       |

PC1, PC3 and PC4 uses only 8 CPU cores for efficiency reasons.

## Back-end
### Starting the cluster
Make sure to edit the IP adresses in the `.env` file (`.env` will create itself from `.env.example` after running the `./start.sh pcX`/`./start_mac.sh` script).

In order to start the cluster use this command in the `PC0` API host:
```bash
./start.sh pc0
```
and in the `PC1` (Worker1):
```bash
./start.sh pc1
```
and in the `PC2` (Worker2):
```bash
./start.sh pc2
```
and in the `PC3` (Worker3) - MacBook has a seperate script because it doesn't use Docker, because Docker can't use Apple Silicone GPU:
```bash
./start_mac.sh
```
and finally the `PC4` (Judge):
```bash
./start.sh pc4
```

## Testing
#### Testing if a `<llm-model>.gguf` can be loaded on individual nodes:
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
./.venv_mac/bin/python3 -c "
from llama_cpp import Llama
print('--- Testing ---')
llm = Llama(model_path='./models/gemma-4-12b-it-Q4_1.gguf', n_gpu_layers=-1, n_ctx=4096)
print('--- Success ---')
"
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
