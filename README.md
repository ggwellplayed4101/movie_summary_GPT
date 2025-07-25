# 🎬 Movie Night Crisis - AI-Powered Interactive Story

An interactive web application that puts you in a social crisis: you've been mistaken for a movie expert and need to convincingly discuss films you've never seen! Powered by a custom-trained GPT model that generates creative movie plot summaries to save your reputation.

![Movie Night Demo](https://img.shields.io/badge/Status-Live%20Demo-green) ![GPT Model](https://img.shields.io/badge/Model-19.17M%20Parameters-blue) ![Training Data](https://img.shields.io/badge/Training%20Data-360MB+-orange)

## 🎯 What This Project Does

You're at a friend's place after movie night. As you're leaving, they mention that someone told them you're a huge movie buff with amazing taste. Problem: you barely watch movies!

The app simulates this awkward conversation where you must:

1. **Recommend movies** on the spot
2. **Explain their plots** convincingly
3. **Save face** using AI-generated plot summaries

## 🚀 Live Demo Features

- **Interactive Storytelling**: Realistic conversation flow with branching dialogue
- **AI Plot Generation**: Custom-trained GPT model creates movie plot summaries
- **Real-time Typing**: Typewriter effect for immersive conversation
- **Responsive Design**: Works on desktop and mobile
- **Smart Fallbacks**: Graceful handling when AI API is unavailable

## 📁 Project Structure

````
movie_night/
├── frontend/                # Interactive web application
│   ├── index.html           # Main HTML structure
│   ├── styles.css           # Responsive CSS styling
│   └── script.js            # Game logic + AI integration
├── backend/                 # nanoGPT-based AI model
│   ├── model.py             # GPT transformer architecture
│   ├── train.py             # Training loop
│   ├── sample.py            # Text generation
│   ├── api_server.py        # Flask API server
│   ├── start_api.py         # API startup script
│   ├── data/shows/          # Training dataset
│   │   ├── input.txt        # Movie/TV show plots (360MB+)
│   │   └── prepare.py       # Data preprocessing
│   └── out-shows/
│       └── ckpt.pt           # Trained model (230MB, 19.17M params)

## 🔥 Technical Highlights

### AI Model Specs

- **Architecture**: GPT-2 style transformer
- **Parameters**: 19.17M (efficient for inference)
- **Training Data**: 360MB+ of movie and TV show plot summaries
- **Context Length**: 1024 tokens
- **Layers**: 12 transformer blocks, 12 attention heads

### Frontend Tech

- **Pure Vanilla JS**: No frameworks, lightweight and fast
- **Modern CSS**: Gradients, animations, responsive design
- **API Integration**: Fetch API with graceful fallbacks
- **Real-time Effects**: Typing animations, loading indicators

### Backend Infrastructure

- **Flask API**: RESTful endpoint for plot generation
- **PyTorch**: Model inference with CPU/GPU support
- **CORS Enabled**: Frontend can call API from different origins
- **Error Handling**: Robust fallbacks and timeout management

## 🛠️ Quick Start

### 1. Frontend Only (Immediate Demo)

```bash
# Open in browser
cd frontend
# Double-click index.html or use a local server
python -m http.server 8080
````

### 2. Full Stack with AI

```bash
# Backend setup
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Start AI API server
python ./api-server.py

# Frontend (in new terminal)
cd frontend
python -m http.server 3000
```

### 3. Test the AI Model

```bash
cd backend
python sample.py --start="A mysterious thriller about" --num_samples=3
```

## 🎮 How to Play

1. **Enter Names**: Your name and your movie-loving friend's name
2. **Get Caught**: The friend asks you to recommend movies
3. **Think Fast**: Enter any movie title when prompted
4. **Watch Magic**: AI generates a convincing plot summary
5. **Survive**: Complete 3 movie recommendations to win!

## 🤖 AI Model Training

The GPT model was trained specifically on movie and TV show plot data:

```bash
# Prepare training data
cd ./backend/
python python train.py config/train_shows.py --device={cpu, cuda, mps, etc} --compile=False --eval_iters=20 --log_interval=1 --block_size=128 --batch_size=32 --n_layer=8 --n_head=8 --n_embd=256 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0

# Train the model (took ~2 hours on GPU)
cd ../../backend
python train.py config/train_shows.py

# Monitor training
# Loss: Started at ~4.2, converged to ~1.8
# Model size: 19.17M parameters
# Final checkpoint: 230MB
```

## 🔧 API Endpoints

### Generate Plot Summary

```bash
POST http://localhost:8000/generate
Content-Type: application/json

{
  "start": "{Movie Title}",
  "max_new_tokens": 150,
  "temperature": 0.8,
  "top_k": 200,
  "num_samples": 1
}
```

### Response Format

```json
{
  "generated_texts": ["AI generated plot summary..."],
  "inference_time": 0.523,
  "model_info": "19.17M parameters"
}
```

## 🎨 Customization

### Modify Story Flow

Edit `script.js` - the `generateStory()` function contains all dialogue:

```javascript
{
  speaker: 'friend',
  text: 'Your custom dialogue here...',
  needsInput: true,
  inputPrompt: 'Custom prompt:'
}
```

### Adjust AI Behavior

Modify API parameters in `script.js`:

```javascript
const AI_API_CONFIG = {
  baseUrl: "http://localhost:8000",
  generateEndpoint: "/generate",
};
```

### Train on Your Data

Replace `data/shows/input.txt` with your own text data and retrain:

```bash
python data/shows/prepare.py
python train.py config/train_shows.py
```

## 🐛 Troubleshooting

### Common Issues

- **AI API not responding**: Check if `start_api.py` is running
- **CORS errors**: Ensure frontend and backend are on correct ports
- **Model loading fails**: Verify `out-shows/ckpt.pt` exists (230MB)
- **Frontend not loading**: Try `python -m http.server` instead of file://

### Debug Mode

Enable verbose logging:

```bash
python start_api.py --debug
```

## 🚀 What's Next

- [ ] **Mobile App**: React Native version
- [ ] **More Scenarios**: Different social situations
- [ ] **Voice Interface**: Speech-to-text integration
- [ ] **Multiplayer**: Multiple friends asking questions
- [ ] **Genre Selection**: Comedy, horror, sci-fi specific plots
- [ ] **Real Movie API**: Integration with TMDB/IMDB

## 🏆 Project Achievements

✅ **Fully Functional AI**: Custom-trained GPT generating creative plots  
✅ **Professional Frontend**: Polished UI with animations  
✅ **API Integration**: Seamless backend communication  
✅ **Mobile Responsive**: Works on all devices  
✅ **Deployment Ready**: Multiple hosting options  
✅ **Well Documented**: Comprehensive guides

## 📝 License

This project is released under the **Creative Commons BY-SA license** to comply with the CMU Movie Summary Corpus licensing requirements.

- You are free to use, modify, and distribute this project
- You must provide attribution to both this project and the original CMU Movie Summary Corpus
- Any derivative works must be shared under the same CC BY-SA license

For more details, see the original dataset license at: http://www.cs.cmu.edu/~dbamman/movies/

## 🙏 Acknowledgments

- **nanoGPT**: Built on Andrej Karpathy's excellent transformer implementation
- **Training Data**: This project uses the **CMU Movie Summary Corpus** by David Bamman, Brendan O'Connor, and Noah A. Smith, released under the Creative Commons BY-SA license. Original dataset available at: https://www.cs.cmu.edu/~ark/personas/
- **Design Inspiration**: Modern web app UX patterns
- **Community**: OpenAI GPT research and PyTorch ecosystem

---

**Built with ❤️ and AI** - A showcase of modern web development meets machine learning
