function loadChatHistory() {
    const chatBox = document.getElementById('chat-messages');
    const history = JSON.parse(localStorage.getItem('chat_history') || '[]');
    history.forEach(line => {
        chatBox.innerHTML += `<div>${line}</div>`;
    });
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;

    const chatBox = document.getElementById('chat-messages');
    chatBox.innerHTML += `<div><strong>You:</strong> ${message}</div>`;
    input.value = '';

    let chatHistory = JSON.parse(localStorage.getItem('chat_history') || '[]');
    chatHistory.push(`<strong>You:</strong> ${message}`);
    localStorage.setItem('chat_history', JSON.stringify(chatHistory));

    const lastSuggestions = localStorage.getItem('last_suggestions') || '';

    try {
        const response = await fetch('/ask_llm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: `The following project suggestions were provided earlier: ${lastSuggestions}. The user now says: ${message}`
            })
        });
        const data = await response.json();
        chatBox.innerHTML += `<div><strong>AI:</strong> ${data.reply}</div>`;

        chatHistory.push(`<strong>AI:</strong> ${data.reply}`);
        localStorage.setItem('chat_history', JSON.stringify(chatHistory));

        chatBox.scrollTop = chatBox.scrollHeight;
    } catch (err) {
        chatBox.innerHTML += `<div><strong>Error:</strong> Could not reach AI service.</div>`;
    }
}

window.onload = loadChatHistory;
