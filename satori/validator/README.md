# Validator

Validator is the validation service of the Satori network, responsible for evaluating model quality, validating datasets, processing audit tasks, and synchronizing scores to the Bittensor network.

## Features

- **Model Quality Evaluation**: CLIP-based aesthetic scoring, NSFW detection
- **Dataset Validation**: Verify miner-submitted dataset format and quality
- **Audit Task Processing**: Receive and process audit tasks from Task Center
- **Weight Synchronization**: Sync scoring results to Bittensor network
- **Bittensor Integration**: Sync metagraph, set weights

## Evaluation Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Receive Task │────▶│ Download    │────▶│ Generate    │
│              │     │ Model       │     │ Samples     │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Submit      │◀────│ Calculate   │◀────│ Quality     │
│ Score       │     │ Score       │     │ Evaluation  │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Installation

### 1. Install Dependencies

```bash
cd satori
pip install -r requirements.txt
pip install -e .
```

### 2. Configuration File

```bash
cp validator/config.example.yml validator/config.yml
```

Edit `config.yml`:

```yaml
wallet:
  name: validator
  hotkey: default

bittensor:
  netuid: 119
  chain_endpoint: wss://entrypoint-finney.opentensor.ai:443

task_center:
  url: http://task.satorilab.ai

validator:
  sync_interval: 60
  validators_per_task: 3
  weight_sync_interval: 1800

models:
  clip:
    model_name: ViT-L/14
    device: cuda

  text_encoder:
    model_name: paraphrase-multilingual-MiniLM-L12-v2

  aesthetic:
    model_name: openai/clip-vit-large-patch14

  nsfw:
    model_name: Falconsai/nsfw_image_detection

scoring:
  k: 3
  baseline: 3.5
  time_decay_rate: 0.005

logging:
  level: INFO
  file: logs/validator.log
```

## Wallet Configuration

Validators require a Bittensor wallet for authentication, signing, and receiving rewards. The wallet must be registered on the subnet with sufficient stake.

### Step 1: Install Bittensor CLI

```bash
pip install bittensor
```

### Step 2: Create Coldkey

The coldkey is your main wallet that holds your TAO tokens.

```bash
btcli wallet new_coldkey --wallet.name validator
```

You will be prompted to:
1. Enter a password (remember this password!)
2. Save the mnemonic phrase (12 or 24 words) - **KEEP THIS SAFE!**

### Step 3: Create Hotkey

The hotkey is used for day-to-day operations, signing audit results, and setting weights.

```bash
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

### Step 4: Fund Your Wallet

Transfer TAO to your coldkey address for staking:

```bash
# Get your coldkey address
btcli wallet overview --wallet.name validator
```

### Step 5: Register on Subnet

Register your validator on the Satori subnet (requires TAO for registration fee):

```bash
btcli subnets register --wallet.name validator --wallet.hotkey default --netuid 119
```

### Step 6: Stake TAO

Stake TAO to increase your validator's weight and earn more rewards:

```bash
# Check current stake
btcli stake list --wallet.name validator

# Add stake (minimum recommended: 1000 TAO)
btcli stake add --wallet.name validator --netuid 119 --amount 1000
```

### Step 7: Verify Registration

```bash
# Check subnet registration
btcli subnets show --netuid 119

# View your validator info
btcli wallet overview --wallet.name validator
```

### Step 8: Update Configuration

Edit `config.yml` to match your wallet names:

```yaml
wallet:
  name: validator        # Your coldkey name
  hotkey: default        # Your hotkey name
```

### Wallet Directory Structure

```
~/.bittensor/wallets/
└── validator/
    ├── coldkey              # Encrypted coldkey (password protected)
    ├── coldkeypub.txt       # Public coldkey
    └── hotkeys/
        └── default          # Hotkey file
```

## Starting the Service

### Option 1: Direct Start

```bash
# Run from project root
python -m satori.validator.validator_main
```

### Option 2: Using PM2

```bash
# Start with PM2
pm2 start "python -m satori.validator.validator_main" \
  --name satori-validator \
  --cwd /path/to/satori-sn119

# Save PM2 process list
pm2 save

# View logs
pm2 logs satori-validator

# Monitor
pm2 monit

# Restart
pm2 restart satori-validator

# Stop
pm2 stop satori-validator
```

## Recommended Hardware

| Component | Recommended                 |
|-----------|-----------------------------|
| GPU | NVIDIA RTX 4080 (16GB VRAM) |
| CPU | 8 cores                     |
| RAM | 32GB                        |
| Storage | 1TB NVMe SSD                |
| Network | 1Gbps                       |

## Weight Synchronization

Validators periodically sync scores to the Bittensor network:

```yaml
validator:
  weight_sync_interval: 1800  # Sync every hour
```

The synced weights affect miner TAO reward distribution.

## Network Configuration

```yaml
bittensor:
  netuid: 119
  chain_endpoint: wss://entrypoint-finney.opentensor.ai:443

axon:
  enabled: true
  ip: 0.0.0.0
  port: 8002
  external_ip: <YOUR_PUBLIC_IP>
```

> **Important Notice:**
> - Each UID is limited to **one external IP address only**.
> - Please open port **8002** (or the port you configured) on your firewall to ensure your validator can properly communicate with the network.
> - Make sure your `external_ip` is set to your public IP address for proper network communication.
