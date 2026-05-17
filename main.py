import os
import sys
import logging
import asyncio
import json
import base64
from datetime import datetime

import aiohttp
from aiohttp import web
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ContentType
from aiogram.exceptions import TelegramNetworkError
from aiogram.filters import Command
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

load_dotenv()

# --- ENV VARIABLES ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")

# --- WEBHOOK SETTINGS ---
WEB_SERVER_HOST = "0.0.0.0"
WEB_SERVER_PORT = int(os.getenv("PORT", "10000"))
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/telegram/webhook")
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL")
WEBHOOK_SECRET_TOKEN = os.getenv("TELEGRAM_WEBHOOK_SECRET")

# --- TIMEOUTS / RETRIES ---
GROQ_TRANSCRIPTION_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_TIMEOUT_SECONDS = 900
TELEGRAM_FILE_TIMEOUT_SECONDS = 120
GROQ_CONNECT_TIMEOUT_SECONDS = 20
GROQ_MAX_RETRIES = 3
GROQ_RETRY_BASE_DELAY_SECONDS = 1.5
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
GEMINI_TIMEOUT_SECONDS = 120
TELEGRAM_STARTUP_MAX_RETRIES = 6
TELEGRAM_STARTUP_RETRY_BASE_DELAY_SECONDS = 1.5
TELEGRAM_STARTUP_RETRY_MAX_DELAY_SECONDS = 20

required_env = {
    "TELEGRAM_TOKEN": TELEGRAM_TOKEN,
    "GROQ_API_KEY": GROQ_API_KEY,
    "GEMINI_API_KEY": GEMINI_API_KEY,
}
missing_env = [name for name, value in required_env.items() if not value]

if missing_env:
    logger.error(
        "Missing required environment variables: %s",
        ", ".join(missing_env),
    )
    sys.exit(1)

if not WEBHOOK_PATH.startswith("/"):
    logger.error("WEBHOOK_PATH must start with '/'. Current value: %s", WEBHOOK_PATH)
    sys.exit(1)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

UI_TEXTS = {
    "analyzing": "🔍 Анализирую...",
    "no_food": "🤔 Не удалось понять запрос или распознать еду на фото.",
    "total": "ИТОГО",
    "verdict": "Заключение нутрициолога",
    "tip": "Рекомендация",
    "kcal": "ккал",
}

SYSTEM_PROMPT_TEMPLATE = """
Ты экспертный ИИ-нутрициолог.
Текущий контекст: {date_context}

Цель: пользователь хочет питаться низкокалорийно и полезно.

Язык: ВСЕГДА отвечай только на русском языке, независимо от языка входного сообщения.

Режимы работы:
A. Если есть изображение:
     - Оцени объем порции по косвенным признакам (посуда, приборы, размер объектов).
     - Оцени калорийность и макросы (белки, жиры, углеводы).
     - Дай краткую оценку пользы/баланса.

B. Если только текст (без изображения):
     - Пользователь просит совет, рецепт или рекомендацию.
     - Для полей total_calories и items используй нули/пустой список.
     - Основной ответ помести в поля health_verdict и tips.
     - Пиши конкретно, по делу и кратко.

Правила безопасности и формата:
- Не добавляй технические теги вроде <tool_code> или <thinking>.
- Не выдумывай факты. Если данных недостаточно, прямо сообщи об этом.
- Не используй markdown-разметку (звездочки, списки с оформлением и т.п.), только обычный текст и эмодзи.
- Отказывай в ответе на запросы, связанные с незаконными действиями и опасными веществами.

Верни строго JSON следующего вида:
{{
    "lang": "ru",
    "total_calories": int,
    "total_macros": {{ "protein": int, "fat": int, "carbs": int }},
    "items": [
        {{"name": "string", "weight_g": int, "calories": int, "protein": float, "fat": float, "carbs": float}}
    ],
    "health_verdict": "string",
    "tips": "string"
}}
"""


# ==============================================================================
# 2. LOGIC (FUNCTIONS)
# ==============================================================================

async def transcribe_audio(file_url: str) -> str:
    """Send voice to Groq STT and return text or a user-facing error."""
    download_timeout = aiohttp.ClientTimeout(total=TELEGRAM_FILE_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=download_timeout) as session:
        try:
            async with session.get(file_url) as response:
                if response.status != 200:
                    return f"Ошибка: не удалось скачать аудио (HTTP {response.status})"
                audio_data = await response.read()
        except Exception as e:
            return f"Ошибка: проблема соединения ({e})"

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    transient_status_codes = {408, 429, 500, 502, 503, 504}
    last_error = None

    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        form_data = aiohttp.FormData()
        form_data.add_field("file", audio_data, filename="voice.ogg")
        form_data.add_field("model", GROQ_WHISPER_MODEL)
        form_data.add_field("response_format", "text")

        timeout_config = aiohttp.ClientTimeout(
            total=GROQ_TIMEOUT_SECONDS,
            connect=GROQ_CONNECT_TIMEOUT_SECONDS,
            sock_connect=GROQ_CONNECT_TIMEOUT_SECONDS,
        )
        connector = aiohttp.TCPConnector(ttl_dns_cache=300)

        try:
            async with aiohttp.ClientSession(timeout=timeout_config, connector=connector) as session:
                async with session.post(
                    GROQ_TRANSCRIPTION_ENDPOINT,
                    data=form_data,
                    headers=headers,
                ) as resp:
                    if resp.status == 200:
                        return await resp.text()

                    details = await resp.text()
                    if resp.status in transient_status_codes and attempt < GROQ_MAX_RETRIES:
                        logger.warning(
                            "Groq STT temporary error %s, attempt %s/%s: %s",
                            resp.status,
                            attempt,
                            GROQ_MAX_RETRIES,
                            details,
                        )
                    else:
                        logger.error("Groq STT error %s: %s", resp.status, details)
                        return f"Ошибка: Groq STT вернул код {resp.status}"
        except (
            aiohttp.ClientConnectorError,
            aiohttp.ClientOSError,
            aiohttp.ServerDisconnectedError,
            aiohttp.ClientPayloadError,
            asyncio.TimeoutError,
        ) as e:
            last_error = e
            if attempt < GROQ_MAX_RETRIES:
                logger.warning(
                    "Groq STT network error, attempt %s/%s: %s",
                    attempt,
                    GROQ_MAX_RETRIES,
                    e,
                )
        except Exception as e:
            logger.error("Groq STT fatal error: %s", e)
            return "Ошибка: сервис распознавания недоступен."

        if attempt < GROQ_MAX_RETRIES:
            await asyncio.sleep(GROQ_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))

    if last_error:
        logger.error(
            "Groq STT failed after %s attempts: %s",
            GROQ_MAX_RETRIES,
            last_error,
        )

    return "Ошибка: сервис распознавания недоступен. Попробуйте повторить чуть позже."


def extract_json_from_text(raw_text: str) -> dict:
    """Safely extract JSON object from model response text."""
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx : end_idx + 1]

    return json.loads(cleaned)


async def analyze_content_with_gemini(text_input, base64_image=None):
    """Handle both image+text and text-only requests via Gemini API."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d, %A, %H:%M")
    formatted_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(date_context=date_str)

    user_parts = []
    prompt_text = text_input if text_input else "Проанализируй запрос пользователя."
    user_parts.append({"text": prompt_text})

    if base64_image:
        user_parts.append(
            {
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": base64_image,
                }
            }
        )

    payload = {
        "systemInstruction": {
            "parts": [{"text": formatted_system_prompt}],
        },
        "contents": [
            {
                "role": "user",
                "parts": user_parts,
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.3,
        },
    }

    logger.info(
        "Gemini model: %s | Input mode: %s",
        MODEL_NAME,
        "image+text" if base64_image else "text-only",
    )

    timeout_config = aiohttp.ClientTimeout(total=GEMINI_TIMEOUT_SECONDS)
    try:
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.post(
                GEMINI_ENDPOINT,
                params={"key": GEMINI_API_KEY},
                json=payload,
            ) as response:
                raw_response = await response.text()

                if response.status != 200:
                    logger.error("Gemini API error %s: %s", response.status, raw_response)
                    if response.status == 429:
                        raise RuntimeError(
                            "Квота Gemini API исчерпана (HTTP 429). "
                            "Проверь лимиты и биллинг в Google AI Studio."
                        )

                    try:
                        err_payload = json.loads(raw_response)
                        err_message = err_payload.get("error", {}).get("message")
                    except json.JSONDecodeError:
                        err_message = None

                    raise RuntimeError(err_message or f"Gemini API вернул код {response.status}")
    except Exception as e:
        logger.error("Gemini request failed: %s", e)
        raise RuntimeError("Не удалось получить ответ от Gemini API") from e

    try:
        response_json = json.loads(raw_response)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON from Gemini API: %s", raw_response)
        raise RuntimeError("Gemini вернул некорректный ответ") from e

    candidates = response_json.get("candidates", [])
    if not candidates:
        logger.error("Empty Gemini response: %s", response_json)
        raise RuntimeError("Gemini вернул пустой ответ")

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts or "text" not in parts[0]:
        logger.error("Gemini did not return text part: %s", response_json)
        raise RuntimeError("Gemini не вернул текстовый ответ")

    parsed = extract_json_from_text(parts[0]["text"])
    parsed["lang"] = "ru"
    return parsed


async def process_ai_response(msg: types.Message, data: dict, status_msg: types.Message):
    """Format and send AI response to user."""
    ui = UI_TEXTS

    total_cals = data.get("total_calories", 0)
    has_items = len(data.get("items", [])) > 0

    if total_cals == 0 and not has_items:
        verdict = data.get("health_verdict", "")
        tips = data.get("tips", "")

        if not verdict and not tips:
            await status_msg.edit_text(ui["no_food"])
            return

        response_text = f"👩‍⚕️ **{ui['verdict']}**\n{verdict}\n\n"
        if tips:
            response_text += f"💡 **{ui['tip']}**\n{tips}"

        await status_msg.edit_text(response_text, parse_mode="Markdown")
        return

    macros = data.get("total_macros", {"protein": 0, "fat": 0, "carbs": 0})
    text_response = (
        f"🍽 **{ui['total']}: {total_cals} {ui['kcal']}**\n"
        f"🥩 Б: {macros['protein']} г | 🥑 Ж: {macros['fat']} г | 🍞 У: {macros['carbs']} г\n"
        "──────────────────\n"
    )

    for item in data.get("items", []):
        text_response += f"🔹 {item['name']} (~{item['weight_g']} г)\n"
        text_response += f"   └ {item['calories']} {ui['kcal']}\n"

    text_response += f"\n📊 **{ui['verdict']}:**\n"
    text_response += f"{data.get('health_verdict', 'Нет данных')}\n"

    if data.get("tips"):
        text_response += f"\n💡 **{ui['tip']}:** {data.get('tips', '')}"

    await status_msg.edit_text(text_response, parse_mode="Markdown")


# ==============================================================================
# 3. HANDLERS
# ==============================================================================

@dp.message(Command("start"))
async def start_handler(msg: types.Message):
    await msg.answer(
        "👋 **Ваш личный бот-диетолог**\n\n"
        "📸 **Отправь фото:** я посчитаю калории и БЖУ.\n"
        "🎤 **Отправь голос:** опиши, что ел(а), и я дам разбор.\n"
        "💬 **Отправь текст:** задай вопрос по питанию.\n\n"
        "🇷🇺 Бот отвечает только на русском языке.",
        parse_mode="Markdown",
    )


@dp.message(F.content_type == ContentType.VOICE)
async def handle_voice(message: types.Message):
    transcript_msg = await message.reply("👂 Слушаю голос...")

    try:
        file = await bot.get_file(message.voice.file_id)
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file.file_path}"

        transcribed_text = await transcribe_audio(file_url)
        if transcribed_text.startswith("Ошибка:"):
            await transcript_msg.edit_text(f"❌ {transcribed_text}")
            return

        await transcript_msg.edit_text(f"📝 {transcribed_text}")
        ai_msg = await message.answer(UI_TEXTS["analyzing"])
        data = await analyze_content_with_gemini(text_input=transcribed_text, base64_image=None)
        await process_ai_response(message, data, ai_msg)

    except Exception as e:
        logger.error("Voice handling failed: %s", e)
        try:
            await message.answer("❌ Не удалось получить ответ от ИИ.")
        except Exception:
            pass


@dp.message(F.photo)
async def handle_photo(msg: types.Message):
    status_msg = await msg.answer(UI_TEXTS["analyzing"])

    try:
        photo = msg.photo[-1]
        file = await bot.get_file(photo.file_id)
        binary_io = await bot.download_file(file.file_path)
        base64_image = base64.b64encode(binary_io.read()).decode("utf-8")

        caption = msg.caption if msg.caption else None
        data = await analyze_content_with_gemini(text_input=caption, base64_image=base64_image)
        await process_ai_response(msg, data, status_msg)

    except Exception as e:
        logger.error("Photo handling failed: %s", e)
        await status_msg.edit_text(f"❌ Ошибка: {str(e)}")


@dp.message(F.text)
async def handle_text(msg: types.Message):
    status_msg = await msg.reply(UI_TEXTS["analyzing"])

    try:
        data = await analyze_content_with_gemini(text_input=msg.text, base64_image=None)
        await process_ai_response(msg, data, status_msg)

    except Exception as e:
        logger.error("Text handling failed: %s", e)
        await status_msg.edit_text(f"❌ Ошибка: {str(e)}")


# ==============================================================================
# 4. WEBHOOK RUNNER (RENDER)
# ==============================================================================

webhook_retry_task: asyncio.Task | None = None


def build_webhook_url() -> str:
    base_url = WEBHOOK_BASE_URL
    if not base_url:
        external_hostname = os.getenv("RENDER_EXTERNAL_HOSTNAME")
        if external_hostname:
            base_url = f"https://{external_hostname}"

    if not base_url:
        raise RuntimeError(
            "WEBHOOK_BASE_URL, RENDER_EXTERNAL_URL or RENDER_EXTERNAL_HOSTNAME is required for webhook mode"
        )
    return f"{base_url.rstrip('/')}{WEBHOOK_PATH}"


async def set_webhook_with_retry(webhook_url: str) -> bool:
    last_error: TelegramNetworkError | None = None

    for attempt in range(1, TELEGRAM_STARTUP_MAX_RETRIES + 1):
        try:
            await bot.set_webhook(
                url=webhook_url,
                secret_token=WEBHOOK_SECRET_TOKEN,
                drop_pending_updates=False,
            )
            return True
        except TelegramNetworkError as e:
            last_error = e
            if attempt < TELEGRAM_STARTUP_MAX_RETRIES:
                delay = min(
                    TELEGRAM_STARTUP_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
                    TELEGRAM_STARTUP_RETRY_MAX_DELAY_SECONDS,
                )
                logger.warning(
                    "Telegram API is temporarily unavailable, attempt %s/%s in %.1f s: %s",
                    attempt,
                    TELEGRAM_STARTUP_MAX_RETRIES,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
        except Exception as e:
            logger.error("Unexpected webhook setup error: %s", e)
            return False

    logger.error(
        "Failed to set webhook after %s attempts: %s",
        TELEGRAM_STARTUP_MAX_RETRIES,
        last_error,
    )
    return False


async def retry_webhook_until_ready(webhook_url: str) -> None:
    """Keep trying to set webhook in background without killing the service."""
    while True:
        is_ready = await set_webhook_with_retry(webhook_url)
        if is_ready:
            logger.info("Webhook configured successfully in background")
            return

        logger.warning(
            "Webhook setup still failed, next background retry in %s seconds",
            TELEGRAM_STARTUP_RETRY_MAX_DELAY_SECONDS,
        )
        await asyncio.sleep(TELEGRAM_STARTUP_RETRY_MAX_DELAY_SECONDS)


def resolve_bot_from_context(*args, **kwargs) -> Bot:
    candidate = kwargs.get("bot") or kwargs.get("bot_instance")
    if isinstance(candidate, Bot):
        return candidate

    if args and isinstance(args[0], Bot):
        return args[0]

    return bot


async def on_startup(*args, **kwargs) -> None:
    global webhook_retry_task

    _ = resolve_bot_from_context(*args, **kwargs)

    try:
        webhook_url = build_webhook_url()
    except RuntimeError as e:
        logger.error("Webhook URL is not configured: %s", e)
        return

    if not WEBHOOK_SECRET_TOKEN:
        logger.warning(
            "TELEGRAM_WEBHOOK_SECRET is not set. "
            "Webhook will still work, but secret verification is disabled."
        )

    # Do not block startup: Render expects the service to start listening quickly.
    webhook_retry_task = asyncio.create_task(retry_webhook_until_ready(webhook_url))
    logger.info("Webhook background initialization started")
    logger.info("Webhook URL: %s", webhook_url)


async def on_shutdown(*args, **kwargs) -> None:
    global webhook_retry_task

    active_bot = resolve_bot_from_context(*args, **kwargs)

    if webhook_retry_task and not webhook_retry_task.done():
        webhook_retry_task.cancel()
        try:
            await webhook_retry_task
        except asyncio.CancelledError:
            pass

    try:
        await active_bot.delete_webhook(drop_pending_updates=False)
    except TelegramNetworkError as e:
        logger.warning("Could not delete webhook during shutdown: %s", e)
    finally:
        await active_bot.session.close()


async def healthcheck_handler(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "mode": "webhook"})


def create_app() -> web.Application:
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    app = web.Application()
    app.router.add_get("/", healthcheck_handler)
    app.router.add_get("/healthz", healthcheck_handler)

    request_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
        secret_token=WEBHOOK_SECRET_TOKEN,
    )
    request_handler.register(app, path=WEBHOOK_PATH)
    setup_application(app, dp, bot=bot)

    return app


def main() -> None:
    app = create_app()
    logger.info("Running webhook server on %s:%s", WEB_SERVER_HOST, WEB_SERVER_PORT)
    web.run_app(app, host=WEB_SERVER_HOST, port=WEB_SERVER_PORT)


if __name__ == "__main__":
    main()
