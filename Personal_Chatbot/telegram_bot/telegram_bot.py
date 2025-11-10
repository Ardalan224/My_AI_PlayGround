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
    ConversationHandler,
)

# --- Load secrets & settings ---
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
DIARY_DIR = Path(os.getenv("DIARY_DIR", "./diary")).resolve()

# Safety: ensure diary dir exists and is inside your project (no traversal)
DIARY_DIR.mkdir(parents=True, exist_ok=True)

TZ = ZoneInfo("Europe/Berlin")  # keep filenames consistent with your app

# --- Conversation states ---
CHOOSING_DATE, ENTERING_DATE = range(2)


# --- Helpers ---
def today_filename() -> Path:
    today_str = datetime.now(TZ).strftime("%d.%m.%Y")
    return DIARY_DIR / f"{today_str}.txt"


# --- Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ALLOWED_USER_ID:
        return  # ignore strangers silently
    await update.message.reply_text(
        "Hi! Send me your memory as one message.\n"
        "After sending it, I’ll ask if it belongs to today or another date."
    )


async def save_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Step 1: receive the memory text
    if update.effective_user.id != ALLOWED_USER_ID:
        return

    msg = (update.message.text or "").strip()
    if not msg:
        await update.message.reply_text("Please send plain text.")
        return

    # Store temporarily in user_data
    context.user_data["pending_message"] = msg

    today_str = datetime.now(TZ).strftime("%d.%m.%Y")
    await update.message.reply_text(
        f"Should I save this entry for today ({today_str}) or another date?\n"
        "Reply with 'today' or 'another'."
    )
    return CHOOSING_DATE


async def choose_date(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Step 2: handle user’s choice (today or another)
    choice = (update.message.text or "").strip().lower()
    if choice == "today":
        date_str = datetime.now(TZ).strftime("%d.%m.%Y")
        context.user_data["chosen_date"] = date_str
        return await save_entry(update, context)
    elif choice == "another":
        await update.message.reply_text("Please enter the date in DD.MM.YYYY format:")
        return ENTERING_DATE
    else:
        await update.message.reply_text("Please reply with 'today' or 'another'.")
        return CHOOSING_DATE


async def enter_date(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Step 3: receive custom date input
    date_str = (update.message.text or "").strip()
    try:
        datetime.strptime(date_str, "%d.%m.%Y")  # validate format
    except ValueError:
        await update.message.reply_text("Invalid format. Please use DD.MM.YYYY.")
        return ENTERING_DATE

    context.user_data["chosen_date"] = date_str
    return await save_entry(update, context)


async def save_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Step 4: actually save the entry to file
    msg = context.user_data.pop("pending_message", "")
    date_str = context.user_data.pop("chosen_date", "")
    path = DIARY_DIR / f"{date_str}.txt"

    OVERWRITE_EACH_DAY = True  # set False if you want to append instead
    if OVERWRITE_EACH_DAY:
        path.write_text(msg + "\n", encoding="utf-8")
        await update.message.reply_text(f"Saved entry (overwrote): {path.name}")
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n\n")
        await update.message.reply_text(f"Appended to: {path.name}")

    return ConversationHandler.END


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

    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.TEXT & ~filters.COMMAND, save_memory)],
        states={
            CHOOSING_DATE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, choose_date)
            ],
            ENTERING_DATE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, enter_date)
            ],
        },
        fallbacks=[CommandHandler("cancel", start)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv_handler)
    app.add_handler(MessageHandler(filters.COMMAND, unknown))

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
