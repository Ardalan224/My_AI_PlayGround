import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)

# --- Load secrets & settings ---
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
DIARY_DIR = Path(os.getenv("DIARY_DIR", "./diary")).resolve()

# Safety: ensure diary dir exists and is inside your project (no traversal)
DIARY_DIR.mkdir(parents=True, exist_ok=True)

TZ = ZoneInfo("Europe/Berlin")  # keep filenames consistent with your app


# --- Helpers ---
def today_filename() -> Path:
    today_str = datetime.now(TZ).strftime("%d.%m.%Y")
    return DIARY_DIR / f"{today_str}.txt"


# --- Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ALLOWED_USER_ID:
        return  # ignore strangers silently
    await update.message.reply_text(
        "Hi! Send me today's memory as one message.\n"
        "I'll save it to your diary as DD.MM.YYYY.txt (Europe/Berlin)."
    )


async def save_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Only accept plain text from the allowed user
    if update.effective_user.id != ALLOWED_USER_ID:
        return

    msg = (update.message.text or "").strip()
    if not msg:
        await update.message.reply_text("Please send plain text.")
        return

    path = today_filename()

    # Choose overwrite OR append:
    OVERWRITE_EACH_DAY = (
        True  # set False if you want to append multiple entries per day
    )

    if OVERWRITE_EACH_DAY:
        path.write_text(msg + "\n", encoding="utf-8")
        await update.message.reply_text(f"Saved today's entry (overwrote): {path.name}")
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n\n")
        await update.message.reply_text(f"Appended to: {path.name}")


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ALLOWED_USER_ID:
        return
    await update.message.reply_text("Just send me your memory as plain text.")


# --- Main ---
def main():
    if not BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")
    if not ALLOWED_USER_ID:
        raise RuntimeError("Missing ALLOWED_USER_ID in .env")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    # Only accept TEXT messages, ignore commands/media
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, save_memory))
    app.add_handler(MessageHandler(filters.COMMAND, unknown))

    # Polling = safe: only outbound HTTPS to Telegram
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
