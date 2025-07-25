// Game state
let gameState = {
  characters: {},
  currentStep: 0,
  isPlaying: false,
  isTyping: false,
  userResponses: {},
  moviesDiscussed: 0,
  movieTitles: [],
  typingSpeed: 30
};

// Story data structure - dynamic based on user input
let story = [];

// API configuration for AI model
const AI_API_CONFIG = {
  baseUrl: 'https://movie-summary-gpt.onrender.com',
  generateEndpoint: '/generate'
};

function generateStory() {
  story = [
    {
      speaker: 'narrator',
      text: 'Movie night is over. You\'re putting on your jacket, ready to leave.',
      scene: 'Living room after movie night - dim lighting, empty popcorn bowls, that cozy post-movie atmosphere',
      needsInput: false
    },
    {
      speaker: 'friend',
      text: 'Hey {userName}, before you go...',
      needsInput: false
    },
    {
      speaker: 'friend',
      text: 'I heard from Sarah that you\'re like, this huge movie buff!',
      needsInput: false
    },
    {
      speaker: 'narrator',
      text: 'Your stomach drops. Movie buff? You barely watch anything...',
      needsInput: false
    },
    {
      speaker: 'user',
      text: 'Oh, um... I mean, I don\'t really watch that many—',
      needsInput: false
    },
    {
      speaker: 'friend',
      text: 'Come on, don\'t be modest! She said you have amazing taste.',
      needsInput: false
    },
    {
      speaker: 'narrator',
      text: 'You fidget with your keys. This is awkward.',
      needsInput: false
    },
    {
      speaker: 'friend',
      text: 'So what\'s a movie you\'d totally recommend?',
      needsInput: true,
      inputPrompt: 'Quick! Name any movie:',
      inputType: 'movie'
    },
    {
      speaker: 'user',
      text: '{movie1}',
      needsInput: false,
      isMovieTitle: true
    },
    {
      speaker: 'friend',
      text: 'Oh interesting! What\'s it about?',
      needsInput: false
    },
    {
      speaker: 'narrator',
      text: 'Panic mode. You have no idea...',
      needsInput: false
    },
    {
      speaker: 'user',
      text: 'AI_PLOT_SUMMARY_1',
      needsInput: false,
      isPlotSummary: true,
      movieIndex: 0
    },
    {
      speaker: 'friend',
      text: 'Wow, that sounds incredible!',
      needsInput: false
    },
    {
      speaker: 'friend',
      text: 'Give me another one!',
      needsInput: true,
      inputPrompt: 'Another movie (you\'re sweating):',
      inputType: 'movie'
    },
    {
      speaker: 'user',
      text: '{movie2}',
      needsInput: false,
      isMovieTitle: true
    },
    {
      speaker: 'friend',
      text: 'Never heard of it. Plot?',
      needsInput: false
    },
    {
      speaker: 'narrator',
      text: 'Your palms are sweaty. Think, think...',
      needsInput: false
    },
    {
      speaker: 'user',
      text: 'AI_PLOT_SUMMARY_2',
      needsInput: false,
      isPlotSummary: true,
      movieIndex: 1
    },
    {
      speaker: 'friend',
      text: 'You really do know your stuff!',
      needsInput: false
    },
    {
      speaker: 'friend',
      text: 'One more - blow my mind!',
      needsInput: true,
      inputPrompt: 'Final movie (reputation on the line):',
      inputType: 'movie'
    },
    {
      speaker: 'user',
      text: '{movie3}',
      needsInput: false,
      isMovieTitle: true
    },
    {
      speaker: 'friend',
      text: 'And this one?',
      needsInput: false
    },
    {
      speaker: 'narrator',
      text: 'Last chance to sound convincing...',
      needsInput: false
    },
    {
      speaker: 'user',
      text: 'AI_PLOT_SUMMARY_3',
      needsInput: false,
      isPlotSummary: true,
      movieIndex: 2
    },
    {
      speaker: 'friend',
      text: '{userName}, you\'re amazing! Sarah was totally right.',
      needsInput: false
    },
    {
      speaker: 'narrator',
      text: 'You smile nervously and head for the door.',
      needsInput: false
    },
    {
      speaker: 'narrator',
      text: 'Somehow, you pulled it off...',
      needsInput: false,
      specialActions: ['showCompletionMessage']
    }
  ];
}

function startStory() {
  const userName = document.getElementById('userName').value.trim();
  const friendName = document.getElementById('friendName').value.trim();

  if (!userName || !friendName) {
    alert('Please enter both names to begin the movie night story!');
    return;
  }

  gameState.characters = { userName, friendName };
  generateStory();
  
  document.getElementById('welcomeScreen').style.display = 'none';
  document.getElementById('storyScreen').style.display = 'flex';
  
  setTimeout(() => {
    startAutoplay();
  }, 500);
}

function processText(text) {
  let processedText = text;
  
  // Replace character names
  Object.keys(gameState.characters).forEach(key => {
    const placeholder = `{${key}}`;
    processedText = processedText.replace(new RegExp(placeholder, 'g'), gameState.characters[key]);
  });
  
  // Replace movie titles
  gameState.movieTitles.forEach((title, index) => {
    const placeholder = `{movie${index + 1}}`;
    processedText = processedText.replace(new RegExp(placeholder, 'g'), title);
  });
  
  return processedText;
}

// Function to integrate with AI model
async function generatePlotSummary(movieTitle) {
  try {
    // Create a prompt for AI model
    const prompt = `"${movieTitle}"`;
    
    // Call AI API
    const response = await fetch(`${AI_API_CONFIG.baseUrl}${AI_API_CONFIG.generateEndpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        start: prompt,
        max_new_tokens: 150,
        temperature: 0.8,
        top_k: 200,
        num_samples: 1,
        seed: Math.floor(Math.random() * 10000) // Random seed for variety
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }

    const data = await response.json();
    
    // Extract the generated text and clean it up
    let generatedText = data.generated_texts[0];
    
    // Clean up the response - take only the plot summary part
    const lines = generatedText.split('\n').filter(line => line.trim());
    let plotSummary = lines[0] || generatedText;
    
    // Ensure it's not too long
    if (plotSummary.length > 300) {
      plotSummary = plotSummary.substring(0, 300) + '...';
    }
    
    // If the AI response is too short or seems incomplete, use fallback
    if (plotSummary.length < 50) {
      return getFallbackPlotSummary(movieTitle);
    }
    
    return plotSummary;
    
  } catch (error) {
    console.error('Error generating plot with AI:', error);
    // Fallback to original logic if API fails
    return getFallbackPlotSummary(movieTitle);
  }
}

// Fallback function (your original logic) if AI fails
function getFallbackPlotSummary(movieTitle) {
  const plotSummaries = {
    'inception': 'A mind-bending thriller where a skilled thief who infiltrates people\'s dreams is given the impossible task of planting an idea instead of stealing one. The film explores multiple layers of reality as the team navigates through dreams within dreams, questioning what\'s real and what\'s fabricated.',
    'the matrix': 'A computer hacker discovers that reality as he knows it is actually a simulated world controlled by machines. He joins a rebellion to fight against the artificial intelligence that has enslaved humanity, learning to bend the rules of the matrix to save mankind.',
    'interstellar': 'In a future where Earth is dying, a former NASA pilot leads a secret mission through a wormhole near Saturn to find humanity a new home. The film combines hard science fiction with emotional storytelling as he races against time while experiencing the effects of relativity.',
    'default': `An engaging story that follows compelling characters through unexpected twists and turns. The film explores themes of ${getRandomThemes()} while delivering ${getRandomGenre()} elements that keep viewers captivated from beginning to end.`
  };

  const key = movieTitle.toLowerCase().replace(/[^a-z0-9\s]/g, '').trim();
  return plotSummaries[key] || plotSummaries['default'];
}

function getRandomThemes() {
  const themes = ['love and loss', 'redemption and hope', 'friendship and betrayal', 'good versus evil', 'coming of age', 'sacrifice and heroism'];
  return themes[Math.floor(Math.random() * themes.length)];
}

function getRandomGenre() {
  const genres = ['dramatic', 'thrilling', 'comedic', 'action-packed', 'mysterious', 'romantic'];
  return genres[Math.floor(Math.random() * genres.length)];
}

async function displayStep(stepIndex) {
  if (stepIndex >= story.length) return;

  const step = story[stepIndex];
  const chatContainer = document.getElementById('chatContainer');

  // Add scene description if present
  if (step.scene && stepIndex === 0) {
    const sceneDiv = document.createElement('div');
    sceneDiv.className = 'scene-description';
    sceneDiv.textContent = step.scene;
    chatContainer.appendChild(sceneDiv);
  }

  // Create message element
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message';

  const messageHeader = document.createElement('div');
  messageHeader.className = 'message-header';

  const speakerSpan = document.createElement('span');
  speakerSpan.className = `speaker ${step.speaker}`;
  speakerSpan.textContent = step.speaker === 'user' ? gameState.characters.userName : 
                          step.speaker === 'friend' ? gameState.characters.friendName : 
                          'Narrator';

  const messageText = document.createElement('div');
  messageText.className = 'message-text';

  messageHeader.appendChild(speakerSpan);
  messageDiv.appendChild(messageHeader);
  messageDiv.appendChild(messageText);
  chatContainer.appendChild(messageDiv);

  // Handle special message types
  if (step.isPlotSummary) {
    // Scroll to bottom first
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // Add AI indicator
    const aiIndicator = document.createElement('div');
    aiIndicator.className = 'ai-indicator';
    aiIndicator.textContent = '🤖 AI Generating Plot Summary...';
    chatContainer.appendChild(aiIndicator);
    
    // Scroll to show AI indicator
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Simulate AI processing time with progress
    for (let i = 0; i < 3; i++) {
      await new Promise(resolve => setTimeout(resolve, 500));
      aiIndicator.textContent += ' ●';
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Remove AI indicator
    chatContainer.removeChild(aiIndicator);

    // Generate plot summary using AI model
    const movieTitle = gameState.movieTitles[step.movieIndex];
    const plotSummary = await generatePlotSummary(movieTitle);

    // Create plot summary element
    const plotDiv = document.createElement('div');
    plotDiv.className = 'plot-summary';
    plotDiv.innerHTML = `<strong>📽️ "${movieTitle}" Plot Summary:</strong>${plotSummary}`;
    messageText.appendChild(plotDiv);

    // Scroll to show plot summary
    chatContainer.scrollTop = chatContainer.scrollHeight;

    await typeMessage(messageText, `Well, "${movieTitle}"'s *clears throat nervously* plot goes like this. ${plotSummary}`);
    
    // Final scroll after typing
    chatContainer.scrollTop = chatContainer.scrollHeight;
  } else {
    // Type the message
    await typeMessage(messageText, processText(step.text));
  }

  // Scroll to bottom after message is complete
  chatContainer.scrollTop = chatContainer.scrollHeight;

  // Increment step counter
  gameState.currentStep++;

  // Handle user input - this will stop autoplay AFTER showing the prompt
  if (step.needsInput) {
    showUserInput(step.inputPrompt || 'Your response:', step.inputType);
    // Stop autoplay after showing input prompt
    gameState.isPlaying = false;
    updatePlayButton();
  } else if (step.specialActions) {
    handleSpecialActions(step.specialActions);
  }
}

async function typeMessage(element, text) {
  gameState.isTyping = true;
  element.innerHTML = element.innerHTML || '';

  const cursor = document.createElement('span');
  cursor.className = 'typing-cursor';
  element.appendChild(cursor);

  for (let i = 0; i < text.length; i++) {
    if (!gameState.isTyping) break;
    
    element.insertBefore(document.createTextNode(text[i]), cursor);
    await new Promise(resolve => setTimeout(resolve, gameState.typingSpeed));
    
    // Scroll during typing to keep content visible
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  if (cursor.parentNode) {
    element.removeChild(cursor);
  }
  gameState.isTyping = false;
  
  // Final scroll after typing completes
  const chatContainer = document.getElementById('chatContainer');
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function showUserInput(prompt, inputType = 'text') {
  const container = document.getElementById('userInputContainer');
  const promptElement = document.getElementById('inputPrompt');
  const inputElement = document.getElementById('userInput');

  promptElement.textContent = prompt;
  inputElement.value = '';
  inputElement.placeholder = inputType === 'movie' ? 'Enter a movie title...' : 'Type your response...';
  container.style.display = 'block';
  inputElement.focus();

  // Scroll to bottom
  setTimeout(() => {
    document.getElementById('chatContainer').scrollTop = document.getElementById('chatContainer').scrollHeight;
  }, 100);
}

function submitUserInput() {
  const input = document.getElementById('userInput').value.trim();
  if (!input) {
    alert('Please enter your response!');
    return;
  }

  const currentStep = story[gameState.currentStep - 1];
  
  // Store movie titles
  if (currentStep && currentStep.inputType === 'movie') {
    gameState.movieTitles.push(input);
    gameState.moviesDiscussed++;
    updateMovieCounter();
  }

  // Store user response
  gameState.userResponses[`response${gameState.currentStep}`] = input;

  // Hide input container
  document.getElementById('userInputContainer').style.display = 'none';

  // Auto-resume play if user was playing before
  gameState.isPlaying = true;
  updatePlayButton();
  
  setTimeout(() => {
    continueAutoplay();
  }, 1000);
}

function updateMovieCounter() {
  const counterElement = document.getElementById('movieCounter');
  if (counterElement) {
    counterElement.textContent = `Movies Discussed: ${gameState.moviesDiscussed}/3`;
  }
}

function handleSpecialActions(actions) {
  actions.forEach(action => {
    if (action === 'showCompletionMessage') {
      setTimeout(() => {
        const chatContainer = document.getElementById('chatContainer');
        const completionDiv = document.createElement('div');
        completionDiv.className = 'scene-description';
        completionDiv.innerHTML = '<strong>🎭 Crisis Averted!</strong><br>Your secret is safe... for now. All thanks to the movie plot AI trained by you! 🤖✨';
        chatContainer.appendChild(completionDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        gameState.isPlaying = false;
        updatePlayButton();
      }, 2000);
    }
  });
}

function toggleAutoplay() {
  gameState.isPlaying = !gameState.isPlaying;
  updatePlayButton();

  if (gameState.isPlaying) {
    continueAutoplay();
  }
}

function continueAutoplay() {
  if (!gameState.isPlaying || gameState.currentStep >= story.length) {
    return;
  }

  displayStep(gameState.currentStep).then(() => {
    // Check if the step just displayed needs input
    const currentStep = story[gameState.currentStep - 1];
    
    if (currentStep && currentStep.needsInput) {
      // The input prompt display will handle stopping autoplay
      return;
    }

    if (gameState.isPlaying && gameState.currentStep < story.length) {
      // Wait 2 seconds after typing completes, then continue
      setTimeout(() => {
        continueAutoplay();
      }, 2000);
    }
  });
}

function startAutoplay() {
  gameState.isPlaying = true;
  updatePlayButton();
  continueAutoplay();
}

function updatePlayButton() {
  const btn = document.getElementById('playPauseBtn');
  if (btn) {
    btn.textContent = gameState.isPlaying ? '⏸ Pause' : '▶ Play';
    btn.classList.toggle('active', gameState.isPlaying);
  }
}

function nextStep() {
  if (gameState.currentStep < story.length && !gameState.isTyping) {
    const currentStep = story[gameState.currentStep - 1];
    if (currentStep && currentStep.needsInput && document.getElementById('userInputContainer').style.display !== 'none') {
      return;
    }
    
    displayStep(gameState.currentStep);
  }
}

// Test function to ensure resetStory is accessible
window.resetStory = function() {
  console.log('Reset button clicked!');
  
  const confirmReset = confirm('Are you sure you want to restart the movie night story? All progress will be lost.');
  console.log('Confirm result:', confirmReset);
  
  if (confirmReset) {
    console.log('User confirmed reset, starting reset process...');
    
    // Stop any ongoing typing or animations
    gameState.isTyping = false;
    
    // Complete reset of game state
    gameState = {
      characters: {},
      currentStep: 0,
      isPlaying: false,
      isTyping: false,
      userResponses: {},
      moviesDiscussed: 0,
      movieTitles: [],
      typingSpeed: 30
    };

    // Clear story data
    story = [];
    console.log('Game state and story cleared');
    
    // Get DOM elements
    const storyScreen = document.getElementById('storyScreen');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const chatContainer = document.getElementById('chatContainer');
    const userInputContainer = document.getElementById('userInputContainer');
    const userName = document.getElementById('userName');
    const friendName = document.getElementById('friendName');
    
    console.log('DOM elements found:', {
      storyScreen: !!storyScreen,
      welcomeScreen: !!welcomeScreen,
      chatContainer: !!chatContainer,
      userInputContainer: !!userInputContainer,
      userName: !!userName,
      friendName: !!friendName
    });
    
    // Force hide story screen
    if (storyScreen) {
      storyScreen.style.display = 'none';
      storyScreen.style.visibility = 'hidden';
      console.log('Story screen hidden');
    }
    
    // Force show welcome screen
    if (welcomeScreen) {
      welcomeScreen.style.display = 'flex';
      welcomeScreen.style.visibility = 'visible';
      console.log('Welcome screen shown');
    }
    
    // Clear chat content
    if (chatContainer) {
      chatContainer.innerHTML = '';
      console.log('Chat cleared');
    }
    
    // Hide user input
    if (userInputContainer) {
      userInputContainer.style.display = 'none';
    }
    
    // Clear input fields
    if (userName) {
      userName.value = '';
      console.log('Username cleared');
    }
    if (friendName) {
      friendName.value = '';
      console.log('Friend name cleared');
    }
    
    // Reset UI controls
    try {
      updatePlayButton();
      updateMovieCounter();
      console.log('UI controls updated');
    } catch (error) {
      console.log('Error updating UI controls:', error);
    }
    
    console.log('Story reset complete - should be back at welcome screen');
    
    // Double check the display states
    setTimeout(() => {
      console.log('Final check - Welcome screen display:', welcomeScreen ? welcomeScreen.style.display : 'not found');
      console.log('Final check - Story screen display:', storyScreen ? storyScreen.style.display : 'not found');
    }, 100);
    
  } else {
    console.log('User cancelled reset');
  }
};

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.target.tagName.toLowerCase() === 'input' || e.target.tagName.toLowerCase() === 'textarea') {
    return;
  }

  switch(e.key) {
    case ' ':
      e.preventDefault();
      toggleAutoplay();
      break;
    case 'ArrowRight':
      e.preventDefault();
      nextStep();
      break;
    case 'Escape':
      gameState.isTyping = false;
      break;
  }
});

// Handle Enter key in user input
document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('userInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitUserInput();
    }
  });
});
