"""
API Server for GPT Model Text Generation - Render Deployment
"""
import os
import pickle
from contextlib import nullcontext
from typing import Optional, List
import torch
import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import GPTConfig, GPT
import requests

def download_model():
    model_path = "out-shows/ckpt.pt"
    
    if not os.path.exists(model_path):
        print("Downloading model checkpoint from Google Drive...")
        
        file_id = "1_Hf7XYgfFfqAYUb4Rm5vJ9t9Eh47sJkV"
        os.makedirs("out-shows", exist_ok=True)
        
        download_large_file_from_google_drive(file_id, model_path)
        print("Model downloaded successfully!")
    else:
        print("Model checkpoint already exists.")

def download_large_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(URL, params=params, stream=True)
            break
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

# CALL IT IMMEDIATELY
download_model()

# Initialize FastAPI app
app = FastAPI(title="GPT Model API", description="API for text generation using trained GPT model")

# Add CORS middleware - UPDATE WITH YOUR VERCEL URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://movie-summary-gpt.vercel.app/", 
        "http://localhost:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "*"  # Remove this in production for better security
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add explicit OPTIONS handler for preflight requests
@app.options("/{full_path:path}")
async def options_handler():
    return {"message": "OK"}

# Configuration
MODEL_CONFIG = {
    'out_dir': os.getenv('MODEL_DIR', 'out-shows'),  # Can be overridden via env var
    'device': os.getenv('DEVICE', 'cpu'),  # Railway free tier is CPU, but flexible
    'dtype': os.getenv('DTYPE', 'float32'),  # Good for CPU
    'compile': os.getenv('COMPILE', 'False').lower() == 'true'  # False by default, safer for deployment
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
    """Load the trained model and tokenizer with fallback handling"""
    global model, encode_fn, decode_fn, device, ctx
    
    try:
        # Set device and context
        device = MODEL_CONFIG['device']
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[MODEL_CONFIG['dtype']]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        
        # Load model
        ckpt_path = os.path.join(MODEL_CONFIG['out_dir'], 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            print(f"⚠️  Model checkpoint not found at {ckpt_path}")
            print("🔄 Running in fallback mode - will use hardcoded responses")
            model = None
            return
        
        print(f"📂 Loading model from {ckpt_path}...")
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
            print(f"📝 Loading meta from {meta_path}...")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            encode_fn = lambda s: [stoi[c] for c in s]
            decode_fn = lambda l: ''.join([itos[i] for i in l])
        else:
            print("🔤 No meta.pkl found, assuming GPT-2 encodings...")
            enc = tiktoken.get_encoding("gpt2")
            encode_fn = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode_fn = lambda l: enc.decode(l)
        
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Continuing in fallback mode...")
        model = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("🚀 Starting up the API server...")
    load_model()

@app.get("/")
async def root():
    return {
        "message": "🎬 Movie Night GPT API is running!", 
        "status": "healthy",
        "model_status": "loaded" if model is not None else "fallback_mode"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": MODEL_CONFIG['device'],
        "mode": "AI-powered" if model is not None else "fallback"
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text using the trained model or fallback"""
    
    # Fallback mode if model isn't loaded
    if model is None:
        print("🔄 Using fallback mode for plot generation")
        
        # Extract movie title from the prompt
        prompt_lower = request.start.lower()
        if 'create a compelling movie plot summary for' in prompt_lower:
            # Extract movie title from the prompt
            start_quote = request.start.find('"')
            end_quote = request.start.find('"', start_quote + 1)
            if start_quote != -1 and end_quote != -1:
                movie_title = request.start[start_quote + 1:end_quote]
            else:
                movie_title = "Unknown Movie"
            
            # Generate a creative fallback plot
            fallback_plots = [
                f"An epic tale of {movie_title} that follows unlikely heroes as they navigate through challenging circumstances. The story weaves together themes of friendship, courage, and redemption in unexpected ways.",
                f"{movie_title} presents a gripping narrative where the main characters face their deepest fears and must make impossible choices. The film combines emotional depth with thrilling action sequences.",
                f"A masterful story in {movie_title} that explores the complexity of human relationships against a backdrop of extraordinary events. The narrative builds to a powerful climax that will leave audiences thinking long after the credits roll."
            ]
            
            import random
            selected_plot = random.choice(fallback_plots)
            fallback_response = f"{request.start}\n\n{selected_plot}"
        else:
            fallback_response = f"{request.start}\n\nThis is a creative response generated in fallback mode since the AI model is not available in the current deployment environment."
        
        return GenerationResponse(
            generated_texts=[fallback_response],
            prompt=request.start,
            parameters={
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "num_samples": request.num_samples,
                "seed": request.seed,
                "fallback_mode": True
            }
        )
    
    # Normal AI model generation
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
                "seed": request.seed,
                "fallback_mode": False
            }
        )
    
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_type": "GPT" if model is not None else "Fallback",
        "device": device if device else "N/A",
        "model_loaded": model is not None,
        "config": {
            "out_dir": MODEL_CONFIG['out_dir'],
            "dtype": MODEL_CONFIG['dtype'],
            "compile": MODEL_CONFIG['compile']
        },
        "status": "AI-powered responses" if model is not None else "Fallback creative responses"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
