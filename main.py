import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = "gemini-3.5-flash"
CARD_COUNT = 20
QUIZ_COUNT = 20
MESSAGE_LIMIT = 4000
POLL_QUESTION_LIMIT = 300
POLL_OPTION_LIMIT = 100
POLL_EXPLANATION_LIMIT = 200
CARDS_PATH = Path(__file__).with_name("cards.json")
_cards_lock = asyncio.Lock()


class Card(BaseModel):
    question: str = Field(description="Короткий учебный вопрос")
    answer: str = Field(description="Короткий точный ответ")


class QuizItem(BaseModel):
    quiz_question: str = Field(
        max_length=POLL_QUESTION_LIMIT,
        description="Проверочный вопрос для викторины",
    )
    options: list[str] = Field(
        min_length=4,
        max_length=4,
        description="Ровно 4 разных варианта ответа",
    )
    correct_option_id: int = Field(
        ge=0,
        le=3,
        description="Индекс правильного варианта от 0 до 3",
    )
    explanation: str = Field(
        max_length=POLL_EXPLANATION_LIMIT,
        description="Краткое объяснение правильного ответа",
    )


class StudyPack(BaseModel):
    cards: list[Card] = Field(
        min_length=CARD_COUNT,
        max_length=CARD_COUNT,
        description=f"Ровно {CARD_COUNT} карточек",
    )
    quizzes: list[QuizItem] = Field(
        min_length=QUIZ_COUNT,
        max_length=QUIZ_COUNT,
        description=f"Ровно {QUIZ_COUNT} вопросов викторины",
    )


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
        "По материалу ниже составь учебные карточки и вопросы викторины.\n"
        f"Карточек должно быть ровно {CARD_COUNT}.\n"
        f"Вопросов викторины должно быть ровно {QUIZ_COUNT}.\n"
        "Язык вопроса и ответа совпадает с языком материала.\n"
        "Если прислана тема, раскрой её базовыми фактами. Если конспект — опирайся на него.\n"
        "Вопросы конкретные, ответы короткие.\n"
        "У каждого вопроса викторины ровно 4 разных варианта, и только один из них верный.\n"
        f"quiz_question — не длиннее {POLL_QUESTION_LIMIT} символов.\n"
        f"Каждый вариант — не длиннее {POLL_OPTION_LIMIT} символов.\n"
        f"explanation — не длиннее {POLL_EXPLANATION_LIMIT} символов.\n"
        "correct_option_id — индекс правильного варианта от 0 до 3.\n\n"
        f"Материал:\n{source}"
    )


def fit_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def normalize_pack(pack: StudyPack) -> StudyPack:
    cards = [
        Card(question=card.question.strip(), answer=card.answer.strip())
        for card in pack.cards
    ]
    if len(cards) != CARD_COUNT or any(not card.question or not card.answer for card in cards):
        raise ValueError(f"Нужно ровно {CARD_COUNT} карточек с вопросом и ответом.")

    normalized_quizzes: list[QuizItem] = []
    for q in pack.quizzes:
        options = [fit_text(option, POLL_OPTION_LIMIT) for option in q.options]
        if len(options) != 4 or any(not option for option in options) or len(set(options)) != 4:
            raise ValueError("У каждого вопроса викторины должно быть ровно 4 разных варианта ответа.")
        if not 0 <= q.correct_option_id <= 3:
            raise ValueError("Индекс правильного ответа должен быть от 0 до 3.")

        normalized_quizzes.append(
            QuizItem(
                quiz_question=fit_text(q.quiz_question, POLL_QUESTION_LIMIT),
                options=options,
                correct_option_id=q.correct_option_id,
                explanation=fit_text(q.explanation, POLL_EXPLANATION_LIMIT),
            )
        )

    if len(normalized_quizzes) != QUIZ_COUNT:
        raise ValueError(f"Нужно ровно {QUIZ_COUNT} вопросов викторины.")

    return StudyPack(cards=cards, quizzes=normalized_quizzes)


async def generate_pack(client: genai.Client, source: str) -> StudyPack:
    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=build_prompt(source),
        config=types.GenerateContentConfig(
            temperature=0.4,
            response_mime_type="application/json",
            response_schema=StudyPack,
            system_instruction=(
                f"Ты составляешь {CARD_COUNT} карточек для запоминания и {QUIZ_COUNT} вопросов викторины. "
                "Верни только JSON по заданной схеме."
            ),
        ),
    )
    parsed = response.parsed
    if isinstance(parsed, StudyPack):
        pack = parsed
    elif parsed is not None:
        pack = StudyPack.model_validate(parsed)
    else:
        pack = StudyPack.model_validate_json(response.text or "")
    return normalize_pack(pack)


def format_cards(cards: list[Card]) -> str:
    blocks = [
        f"{index}. {card.question}\n→ {card.answer}"
        for index, card in enumerate(cards, start=1)
    ]
    return "\n\n".join(blocks)


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


def append_study_pack(topic: str, pack: StudyPack, path: Path = CARDS_PATH) -> None:
    if path.exists():
        raw = path.read_text(encoding="utf-8").strip()
        data = json.loads(raw) if raw else []
        if not isinstance(data, list):
            raise ValueError("cards.json должен содержать список наборов.")
    else:
        data = []

    data.append(
        {
            "topic": topic,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "cards": [card.model_dump() for card in pack.cards],
            "quizzes": [quiz.model_dump() for quiz in pack.quizzes],
        }
    )

    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


async def main() -> None:
    bot = Bot(token=get_bot_token())
    client = genai.Client(api_key=get_gemini_api_key())
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def cmd_start(message: Message) -> None:
        name = message.from_user.full_name if message.from_user else "друг"
        await message.answer(
            f"Привет, {name}! Пришли мне тему или конспект лекции для составления карточек и вопросов викторины — "
            f"я пришлю {CARD_COUNT} карточек и {QUIZ_COUNT} вопросов викторины для каждого из них."
        )

    @dp.message(F.text, ~F.text.startswith("/"))
    async def make_cards(message: Message) -> None:
        source = (message.text or "").strip()
        if not source:
            return

        status = await message.answer(
            f"Составляю {CARD_COUNT} карточек и {QUIZ_COUNT} вопросов викторины…"
        )
        try:
            pack = await generate_pack(client, source)
        except Exception:
            logger.exception("Не удалось получить карточки от Gemini")
            await status.edit_text(
                "Не получилось составить карточки. Проверьте ключ Gemini и попробуйте ещё раз."
            )
            return

        saved = True
        try:
            async with _cards_lock:
                await asyncio.to_thread(append_study_pack, source, pack)
        except Exception:
            saved = False
            logger.exception("Не удалось записать cards.json")

        parts = split_text(format_cards(pack.cards))
        await status.edit_text(parts[0])
        for part in parts[1:]:
            await message.answer(part)

        for quiz in pack.quizzes:
            await message.answer_poll(
                question=quiz.quiz_question,
                options=quiz.options,
                type="quiz",
                correct_option_id=quiz.correct_option_id,
                explanation=quiz.explanation or None,
                is_anonymous=False,
            )
            await asyncio.sleep(0.5)

        if not saved:
            await message.answer("Карточки отправлены, но записать cards.json не удалось.")

    logger.info("Бот запущен. Ожидаю сообщения...")
    try:
        await dp.start_polling(bot)
    finally:
        await client.aio.aclose()


if __name__ == "__main__":
    asyncio.run(main())