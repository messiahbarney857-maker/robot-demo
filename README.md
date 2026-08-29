# Robot Demo

This repository contains a local robot demo combining vision, speech, and a ChatGPT-style chat.

## Try the local chat UI

I added a simple web chat interface you can run locally and connect to the ChatGPT API (if you provide a key).

Run locally:

1. Clone the repo and create a virtualenv

   ```bash
   git clone https://github.com/messiahbarney857-maker/robot-demo.git
   cd robot-demo
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   pip install flask python-dotenv
   ```

3. Create a `.env` file with your OpenAI API key (optional — without it the server will run but the LLM replies will be disabled):

   ```env
   OPENAI_API_KEY=sk-...
   ```

4. Start the chat server

   ```bash
   python app.py
   ```

5. Open your browser at http://localhost:5000 and click the "Chat" button to start a conversation.

Note: The web UI is local only. If you want me to deploy it to a public URL (e.g., Render, Heroku, or GitHub Pages + server), tell me and I can add deployment config.
