# MultiAgent Chat-Bot
Ai-Chat bot with Judge-Worker architecture. The project consists of backend + API, PWA app and mobile app.

## Requirements
1. Two PCs preferably with Linux or WSL2
2. Both PCs have to be connected either to the same network or to the same VPN
3. Each having at least 32GB of RAM (less should also work, but it's not recommended)
4. And each having at least one Nvidia GPU with at least 8GB of VRAM (6GB should also work but then you'll have to choose different LLMs - smaller)
5. Both PCs needs to have the CUDA drivers installed
6. Both PCs needs to have Docker

## Back-end
# Starting the cluster
Starting the AI-Cluster requires using this command in the first PC:
```bash
./start.sh pc1
```
and in the second PC:
```bash
./start.sh pc2
```
In our case, the PC1 has more VRAM than the PC2, therefore PC1 is being used for the `Fast` mode, and as a Worker's host in the other modes.

Make sure to edit the IP adresses in the `.env` file.

# Testing
You can test the connection between the PCs and if the models can be loaded by using this commans:
```bash
curl -X POST http://<PC2_IP>:8000/system/init_mode -H "Content-Type: application/json" -d '{"mode": "pro"}'
```
If everything went according to plan, then you should receive the confirmation.
