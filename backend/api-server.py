"""
API Server for GPT Model Text Generation
"""
import os
import pickle
from contextlib import nullcontext
from typing import Optional, List
import torch
import tiktoken
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import GPTConfig, GPT

# Initialize FastAPI app
app = FastAPI(title="GPT Model API", description="API for text generation using trained GPT model")

# Configuration
MODEL_CONFIG = {
    'out_dir': 'out-shows',  # Default model directory
    'device': 'cpu',  # Default to CPU, change to 'cuda' if you have GPU
    'dtype': 'float32',  # Use float32 for CPU
    'compile': False
}

# Global model variables
model = None
encode_fn = None
decode_fn = None
device = None
ctx = None

class GenerationRequest(BaseModel):
    start: str
    max_new_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 200
    num_samples: Optional[int] = 1
    seed: Optional[int] = None

class GenerationResponse(BaseModel):
    generated_texts: List[str]
    prompt: str
    parameters: dict

def load_model():
    """Load the trained model and tokenizer"""
    global model, encode_fn, decode_fn, device, ctx
    
    # Set device and context
    device = MODEL_CONFIG['device']
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[MODEL_CONFIG['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Load model
    ckpt_path = os.path.join(MODEL_CONFIG['out_dir'], 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
    
    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    # Load state dict and clean up keys
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    if MODEL_CONFIG['compile']:
        model = torch.compile(model)
    
    # Load tokenizer
    load_meta = False
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode_fn = lambda s: [stoi[c] for c in s]
        decode_fn = lambda l: ''.join([itos[i] for i in l])
    else:
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode_fn = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode_fn = lambda l: enc.decode(l)
    
    print("Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    return {"message": "GPT Model API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": MODEL_CONFIG['device']
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text using the trained model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Set seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)
            if device != 'cpu':
                torch.cuda.manual_seed(request.seed)
        
        # Handle file input
        start_text = request.start
        if start_text.startswith('FILE:'):
            try:
                with open(start_text[5:], 'r', encoding='utf-8') as f:
                    start_text = f.read()
            except FileNotFoundError:
                raise HTTPException(status_code=400, detail=f"File not found: {start_text[5:]}")
        
        # Encode the prompt
        start_ids = encode_fn(start_text)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        
        generated_texts = []
        
        # Generate samples
        with torch.no_grad():
            with ctx:
                for _ in range(request.num_samples):
                    y = model.generate(
                        x, 
                        request.max_new_tokens, 
                        temperature=request.temperature, 
                        top_k=request.top_k
                    )
                    generated_text = decode_fn(y[0].tolist())
                    generated_texts.append(generated_text)
        
        return GenerationResponse(
            generated_texts=generated_texts,
            prompt=start_text,
            parameters={
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "num_samples": request.num_samples,
                "seed": request.seed
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": "GPT",
        "device": device,
        "config": {
            "out_dir": MODEL_CONFIG['out_dir'],
            "dtype": MODEL_CONFIG['dtype'],
            "compile": MODEL_CONFIG['compile']
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)