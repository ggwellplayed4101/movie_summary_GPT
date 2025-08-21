# 🎬 Movie Night Crisis - AI-Powered Interactive Story

An interactive web application that puts you in a social crisis: you've been mistaken for a movie expert and need to convincingly discuss films you've never seen! Powered by a custom-trained GPT model that generates creative movie plot summaries to save your reputation.

![Movie Night Demo](https://img.shields.io/badge/Status-Live%20Demo-green) ![GPT Model](https://img.shields.io/badge/Model-19.17M%20Parameters-blue) ![Training Data](https://img.shields.io/badge/Training%20Data-360MB+-orange)

## 🎯 What This Project Does

You're at a friend's place after movie night. As you're leaving, they mention that someone told them you're a huge movie buff with amazing taste. Problem: you barely watch movies!

The app simulates this awkward conversation where you must:

1. **Recommend movies** on the spot
2. **Explain their plots** convincingly
3. **Save face** using AI-generated plot summaries

## 🛠️ How We Built It

- **Data**: Trained on the [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/) (~42k movie plots, 360MB), licensed under CC BY-SA  
- **Model**: Custom **nanoGPT** (8 layers, 8 heads, 256 embedding size, ~19M parameters)  
- **Training**: ~2 hours on GPU, loss converged to ~1.8, final checkpoint ~230MB  
- **Integration**: Backend serves AI completions; React frontend simulates the social scenario with dialogue, pacing, and effects  

👉 End result: a custom AI movie buff that invents convincing plots on demand.

## 🎬 Demo

![App Demo](./assets/demo.gif)

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
