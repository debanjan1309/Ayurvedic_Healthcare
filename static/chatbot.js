// Initialize SpeechRecognition
const recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let isListening = false;
let isProcessing = false; // Flag to prevent duplicate submissions

if (recognition) {
    const recognitionInstance = new recognition();
    recognitionInstance.lang = 'en-US';
    recognitionInstance.interimResults = false;
    recognitionInstance.maxAlternatives = 1;

    recognitionInstance.onstart = () => {
        isListening = true;
        document.getElementById('voice-input').innerText = '🔊'; // Change icon or text to indicate listening
    };

    recognitionInstance.onresult = (event) => {
        if (isProcessing) return; // Prevent duplicate processing

        isProcessing = true; // Set the flag to indicate processing
        const speechToText = event.results[0][0].transcript;
        document.getElementById('user-input').value = speechToText;
        addMessage(speechToText, 'user'); // Show the speech input in chat

        // Send the message
        sendMessage(speechToText);

        // Stop recognition after processing the result
        recognitionInstance.stop();
    };

    recognitionInstance.onerror = (event) => {
        console.error('Speech recognition error', event.error);
        isListening = false;
        isProcessing = false; // Reset the flag
        document.getElementById('voice-input').innerText = '🎙️'; // Reset button icon or text
    };

    recognitionInstance.onend = () => {
        isListening = false;
        isProcessing = false; // Reset the flag
        document.getElementById('voice-input').innerText = '🎙️'; // Reset button icon or text
    };

    document.getElementById('voice-input').addEventListener('click', () => {
        if (isListening) {
            recognitionInstance.stop();
        } else {
            recognitionInstance.start();
        }
    });
} else {
    console.log('Speech recognition not supported');
}

document.getElementById('chatbot-icon').addEventListener('click', function() {
    document.getElementById('chatbot-container').style.display = 'flex';
});

document.getElementById('close-chatbot').addEventListener('click', function() {
    document.getElementById('chatbot-container').style.display = 'none';
});

document.getElementById('chat-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const userInput = document.getElementById('user-input').value;
    
    if (userInput.trim() !== '') {
        sendMessage(userInput);
    }
});

function sendMessage(userInput) {
    // Add user message with animation
    addMessage(userInput, 'user');

    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        // Add bot response with animation
        addMessage(data.response, 'bot');
        speakText(data.response); // Optional: Speak the bot's response
        document.getElementById('user-input').value = '';
    });
}

function addMessage(text, type) {
    const chatLog = document.getElementById('chat-log');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.innerHTML = `<div class="text">${text}</div>`;
    chatLog.appendChild(messageDiv);
    chatLog.scrollTop = chatLog.scrollHeight; // Scroll to the bottom
}

// Optional: Function to speak text using the Web Speech API
function speakText(text) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    speechSynthesis.speak(utterance);
}
