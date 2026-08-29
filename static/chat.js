const messagesDiv = document.getElementById('messages');
const msgInput = document.getElementById('msg');
const sendBtn = document.getElementById('send');
const clearBtn = document.getElementById('clear');

function appendMessage(text, cls) {
  const el = document.createElement('div');
  el.className = 'msg ' + cls;
  el.textContent = text;
  messagesDiv.appendChild(el);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

sendBtn.onclick = async () => {
  const text = msgInput.value.trim();
  if (!text) return;
  appendMessage('You: ' + text, 'user');
  msgInput.value = '';
  try {
    const resp = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
    const data = await resp.json();
    if (data.reply) {
      appendMessage('Robot: ' + data.reply, 'bot');
    } else if (data.error) {
      appendMessage('Error: ' + data.error, 'bot');
    }
  } catch (e) {
    appendMessage('Network error: ' + e, 'bot');
  }
}

clearBtn.onclick = () => { messagesDiv.innerHTML = ''; }

// allow Ctrl+Enter to send
msgInput.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey||e.metaKey)) sendBtn.click(); });
