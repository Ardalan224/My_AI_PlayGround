## 🧠 ArdaBrain — Your Private Diary Chatbot

ArdaBrain is a local, privacy-focused chatbot that lets you chat with your personal memories.
It reads your daily text notes from the diary/ folder and uses local LLMs (via Ollama) to recall what you’ve written — no internet or external APIs required.

## 🚀 Quick Start
## 1️⃣ Requirements

Make sure you have:

Python 3.11+

Streamlit

Ollama
 installed and running locally
(used for both embeddings and chat model)

# Install dependencies:

pip install -r requirements.txt

## 2️⃣ Add Your Diary Notes

Place your text files inside the diary/ folder.

File format: DD.MM.YYYY.txt

Example:

diary/
├── 03.02.2025.txt
├── 20.10.2025.txt
└── 24.10.2025.txt


Each file can contain any text you want — thoughts, memories, or daily logs.

## 3️⃣ Run the Chatbot App

Launch the Streamlit interface:

streamlit run app.py


Then open the local URL shown in the terminal (usually http://localhost:8501
).

You can ask questions like:

“What did I do on 03.02.2025?”
“When was the last time I took a bus?”
“Tell me a fun or scary memory.”

Everything runs completely offline.

## 💬 Optional: Add Memories via Telegram

You can also add daily memories from your phone using Telegram.

Setup steps

Create a Telegram Bot using @BotFather

→ Save the API token it gives you.

Create a .env file in the telegram_bot/ folder:

TELEGRAM_TOKEN=YOUR_BOT_TOKEN_HERE
DIARY_DIR=C:/path/to/your/diary


Run the bot:

cd telegram_bot
python bot.py


Open your Telegram app, find your bot, and send it a message.
Every message you send will be saved as a .txt file in your diary folder — named automatically by date (e.g. 24.10.2025.txt).

🛡️ The bot runs locally on your computer — it doesn’t expose your PC to the internet.

🧩 Customization

You can easily change:

The chatbot’s name or greeting (in app.py)

The LLM model tag and temperature in the sidebar

Or even rename the app in the header line:

APP_NAME = "ArdaBrain"

## 📝 Notes

The vector index rebuilds automatically when the app starts.

All embeddings and chat runs stay on your device.

No external API calls, no cloud storage — your memories remain private.