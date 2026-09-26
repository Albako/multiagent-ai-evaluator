# MultiAgent AI Evaluator

## Hardware
My project is hosted on 4 different hosts:
1. PC0 API: My x86 server.
2. PC1 Worker1: Nvidia Jetson AGX Xavier Developer Kit 16GB 4267MT/s with MAXN mode enabled  
3. PC2 Worker2: R9 7945HX 64GB 5200MT/s + RTX 4060 (mobile) 8GB GDDR6
4. PC3 Judge: i7-14700KF 64GB 6400MT/s + RTX 4070 Ti 12GB GDDR6X

## Back-end
### Starting the cluster
Make sure to edit the IP adresses in the `.env` file (`.env` will create itself from `.env.example` after running the `./start.sh pcx` script).

In order to start the cluster use this command in the API host:
```bash
./start.sh pc0
```
in the Nvidia Jeston AGX Xavier PCarm64 (Worker1):
```bash
./start.sh pc1
```
and in the second PCx86 (Worker2):
```bash
./start.sh pc2
```
and finally the third PCx86 (Judge):
```bash
./start.sh pc3
```

### Testing
Start by loading the models:
```bash
curl -X POST "http://127.0.0.1:8000/system/init_mode" \
     -H "Content-Type: application/json" \
     -d '{
           "mode": "coding"
         }'
```
Then test the connection:
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
           "message": "Write a Python function to check if a number is prime and optimize its time complexity.",
           "mode": "coding"
         }'
```
