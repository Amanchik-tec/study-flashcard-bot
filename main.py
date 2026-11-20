import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = "gemini-3.5-flash"
CARD_COUNT = 20
MESSAGE_LIMIT = 4000


def get_bot_token() -> str:
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "BOT_TOKEN не задан. Вставьте токен бота в файл .env и запустите снова."
        )
    return token


def get_gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY не задан. Вставьте ключ Gemini в файл .env и запустите снова."
        )
    return key


def build_prompt(source: str) -> str:
    return (
        "Пользователь прислал тему или конспект. Составь по нему учебные карточки.\n\n"
        f"Нужно ровно {CARD_COUNT} карточек. Каждая карточка — отдельная строка:\n"
        "1. Вопрос — Ответ\n"
        "2. Вопрос — Ответ\n"
        "...\n"
        f"{CARD_COUNT}. Вопрос — Ответ\n\n"
        "Правила:\n"
        f"- ровно {CARD_COUNT} строк, без пропусков в нумерации\n"
        "- между вопросом и ответом стоит длинное тире « — »\n"
        "- без вступления, заголовка и комментариев после списка\n"
        "- язык карточек совпадает с языком материала\n"
        "- вопросы конкретные, ответы короткие и проверяемые\n"
        "- если прислана тема, раскрой её базовыми фактами; если конспект — опирайся на него\n\n"
        "Материал:\n"
        f"{source}"
    )


async def generate_cards(client: genai.Client, source: str) -> str:
    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=build_prompt(source),
        config=types.GenerateContentConfig(
            temperature=0.4,
            system_instruction=(
                "Ты составляешь карточки для запоминания. "
                "Отвечай только нумерованным списком в формате «Вопрос — Ответ»."
            ),
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini вернул пустой ответ.")
    return text


def split_text(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for line in text.splitlines(keepends=True):
        if len(line) > limit:
            if current:
                chunks.append("".join(current).strip())
                current = []
                size = 0
            for start in range(0, len(line), limit):
                part = line[start : start + limit].strip()
                if part:
                    chunks.append(part)
            continue

        if current and size + len(line) > limit:
            chunks.append("".join(current).strip())
            current = [line]
            size = len(line)
        else:
            current.append(line)
            size += len(line)

    if current:
        chunks.append("".join(current).strip())

    return [chunk for chunk in chunks if chunk]


async def main() -> None:
    bot = Bot(token=get_bot_token())
    client = genai.Client(api_key=get_gemini_api_key())
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def cmd_start(message: Message) -> None:
        name = message.from_user.full_name if message.from_user else "друг"
        await message.answer(
            f"Привет, {name}! Пришли тему или конспект — "
            f"я составлю {CARD_COUNT} карточек в формате «Вопрос — Ответ»."
        )

    @dp.message(F.text, ~F.text.startswith("/"))
    async def make_cards(message: Message) -> None:
        source = (message.text or "").strip()
        if not source:
            return

        status = await message.answer(f"Составляю {CARD_COUNT} карточек…")
        try:
            cards = await generate_cards(client, source)
        except Exception:
            logger.exception("Не удалось получить карточки от Gemini")
            await status.edit_text(
                "Не получилось составить карточки. Проверьте ключ Gemini и попробуйте ещё раз."
            )
            return

        parts = split_text(cards)
        await status.edit_text(parts[0])
        for part in parts[1:]:
            await message.answer(part)

    logger.info("Бот запущен. Ожидаю сообщения...")
    try:
        await dp.start_polling(bot)
    finally:
        await client.aio.aclose()


if __name__ == "__main__":
    asyncio.run(main())
