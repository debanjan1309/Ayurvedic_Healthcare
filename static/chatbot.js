document.getElementById('chatbot-icon').addEventListener('click', function() {
    document.getElementById('chatbot-container').style.display = 'flex';
});

document.getElementById('close-chatbot').addEventListener('click', function() {
    document.getElementById('chatbot-container').style.display = 'none';
});

document.getElementById('chat-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const userInput = document.getElementById('user-input').value;
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        const chatLog = document.getElementById('chat-log');
        chatLog.innerHTML += `<p><strong>You:</strong> ${userInput}</p>`;
        chatLog.innerHTML += `<p><strong>Bot:</strong> ${data.response}</p>`;
        document.getElementById('user-input').value = '';
        chatLog.scrollTop = chatLog.scrollHeight;
    });
});
