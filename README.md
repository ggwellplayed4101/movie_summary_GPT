# Movie Night Story + nanoGPT

A full-stack interactive storytelling application that combines a frontend movie conversation game with a GPT transformer model backend for AI-generated plot summaries.

## Project Structure

```
├── frontend/           # Interactive movie night story web app
│   └── index.html     # Complete HTML/CSS/JS application
├── backend/           # nanoGPT transformer model implementation
│   ├── train.py       # Main training script
│   ├── model.py       # GPT model definition
│   ├── sample.py      # Text generation/sampling
│   ├── bench.py       # Performance benchmarking
│   ├── configurator.py # Configuration management
│   └── requirements.txt # Python dependencies
└── README.md          # This file
```

## Frontend - Movie Night Story

An interactive web application where users engage in a conversation about movies. The app simulates a scenario where you need to recommend movies and provide plot summaries on the spot.

### Features

- Character setup with custom names
- Interactive conversation flow
- AI-generated plot summaries (simulated in demo)
- Responsive design with animations
- Auto-play and manual controls

### Usage

Simply open `frontend/index.html` in a web browser to start the interactive story.

## Backend - nanoGPT Implementation

A simplified, fast repository for training/finetuning medium-sized GPT models. Based on Andrej Karpathy's nanoGPT, prioritizing simplicity and readability.

### Key Files

- `train.py` - ~300-line training loop that reproduces GPT-2 (124M) on OpenWebText
- `model.py` - ~300-line GPT model definition with optional GPT-2 weight loading
- `sample.py` - Generate text samples from trained models
- `bench.py` - Simple benchmarking and profiling
- `configurator.py` - Command-line configuration system

### Installation

```bash
cd backend
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Dependencies

- PyTorch >=2.0.0
- NumPy
- tiktoken (OpenAI's BPE tokenizer)
- transformers (HuggingFace, for loading GPT-2 checkpoints)
- wandb (optional logging)

### Quick Start

**Train a character-level GPT on Shakespeare:**

```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
```

**Sample from the trained model:**

```bash
python sample.py --out_dir=out-shakespeare-char
```

**For CPU-only training:**

```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

### Training GPT-2 Scale Models

**Prepare OpenWebText dataset:**

```bash
python data/openwebtext/prepare.py
```

**Train GPT-2 (124M) - requires 8x A100 40GB:**

```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

**Single GPU training:**

```bash
python train.py
```

### Sampling/Inference

Sample from pre-trained GPT-2:

```bash
python sample.py --init_from=gpt2-xl --start="What is the answer to life, the universe, and everything?" --num_samples=5 --max_new_tokens=100
```

Sample from your trained model:

```bash
python sample.py --out_dir=your-model-dir
```

### Integration Potential

The frontend's AI plot generation feature is currently simulated but could be integrated with the trained GPT model for real AI-generated movie plot summaries.

## Performance Notes

- Uses PyTorch 2.0 `torch.compile()` for significant speedup
- Supports distributed training with DDP
- Flash Attention for efficient GPU utilization
- Benchmarking tools included for optimization

## Troubleshooting

- If you encounter PyTorch 2.0 compilation issues, add `--compile=False`
- For Windows compatibility issues, ensure you have the latest PyTorch
- For memory issues, reduce `batch_size`, `block_size`, or model size

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Backend based on Andrej Karpathy's nanoGPT
- Frontend is an original interactive storytelling implementation
