from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import openai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

app = Flask(__name__, static_folder='static', template_folder='templates')

# Simple in-memory chat session
class ChatSession:
    def __init__(self, system_prompt="You are a helpful robot assistant. Keep replies short and friendly."):
        self.system_prompt = system_prompt
        self.messages = [{"role":"system","content":self.system_prompt}]

    def ask(self, user_text):
        if not OPENAI_API_KEY:
            return "No OPENAI_API_KEY set. Set it in a .env file to enable ChatGPT replies."
        self.messages.append({"role":"user","content":user_text})
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                max_tokens=200,
                temperature=0.7
            )
            assistant_text = resp.choices[0].message["content"].strip()
            self.messages.append({"role":"assistant","content":assistant_text})
            return assistant_text
        except Exception as e:
            return f"OpenAI error: {e}"

chat_session = ChatSession()

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json() or {}
    msg = data.get('message','').strip()
    if not msg:
        return jsonify({"error":"empty message"}), 400
    reply = chat_session.ask(msg)
    return jsonify({"reply": reply})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
