# MultiAgent Chat-Bot
Ai-Chat bot with Judge-Worker architecture. The project consists of backend + API, PWA app and mobile app.

## Requirements
1. Four PCs preferably with Linux or WSL2
2. Each PCs have to be connected either to the same network or to the same VPN
3. Three out of four having at least 32GB of RAM (less should also work, but it's not recommended) (exemptions are devices with unified memory +16GB)
4. And each having at least one Nvidia GPU
5. Three out of four PCs needs to have the CUDA drivers installed
6. Three out of four PCs needs to have Docker

## Back-end
### Starting the cluster
In order to start the cluster use this command in the weakest PC (the one without GPU):
```bash
./start.sh pc0
```
in the first PC:
```bash
./start.sh pc1
```
and in the second PC:
```bash
./start.sh pc2
```
and finally the third:
```bash
./start.sh pc3
```

Make sure to edit the IP adresses in the `.env` file.

### Testing
You can test the connection between the PCs and if the models can be loaded by using this commans:
```bash
curl -X POST http://<PC2_IP>:8000/system/init_mode -H "Content-Type: application/json" -d '{"mode": "pro"}'
```
If everything went according to plan, then you should receive the confirmation.
