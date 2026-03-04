# MultiAgent Chat-Bot
Ai-Chat bot with Judge-Worker architecture. The project consists of backend + API, PWA app and mobile app.

## Requirements
1. Two PCs preferably with Linux or WSL2
2. Both PCs have to be connected either to the same network or to the same VPN
3. Each having at least 32GB of RAM
4. And each having at least 1 Nvidia GPU with at least 8GB of VRAM (6GB should also work but then you'll have to choose different LLMs - smaller)

## Back-end
Starting the Back-end requires using this command in the first PC:
```bash
./start.sh pc1
```
and in the second PC:
```bash
./start.sh pc2
```
