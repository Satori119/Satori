# Task Center

Task Center is the core coordination service of the Satori network, responsible for task lifecycle management, miner selection, audit task distribution, and reward calculation.

## Installation

### 1. Install Dependencies

```bash
cd satori
pip install -r requirements.txt
pip install -e .
```

### 2. Database Setup

```bash
# Create PostgreSQL database
sudo -u postgres psql -c "CREATE USER satori WITH PASSWORD 'satori';"
sudo -u postgres psql -c "CREATE DATABASE satori OWNER satori;"
```

### 3. Configuration File

```bash
cp task_center/config.example.yml task_center/config.yml
```

Edit `config.yml`:

```yaml
database:
  url: postgresql://satori:satori@localhost:5432/satori

bittensor:
  netuid: 119
  chain_endpoint: wss://entrypoint-finney.opentensor.ai:443

logging:
  level: INFO
  file: logs/task_center.log
```

## Starting the Service

### Option 1: Direct Start

```bash
# Run from project root
python -m satori.task_center.task_center_main
```

### Option 2: Using PM2

```bash
# Start with PM2
pm2 start "python -m satori.task_center.task_center_main" \
  --name satori-task-center \
  --cwd /path/to/satori-sn119

# Save PM2 process list
pm2 save

# View logs
pm2 logs satori-task-center

# Monitor
pm2 monit

# Restart
pm2 restart satori-task-center

# Stop
pm2 stop satori-task-center
```

### Option 3: Using uvicorn

```bash
# Note: Must use --loop asyncio because bittensor doesn't support uvloop
uvicorn satori.task_center.task_center_main:app --host 0.0.0.0 --port 8000 --loop asyncio
```