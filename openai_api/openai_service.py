import time

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pathlib import Path
from pydantic import BaseModel
from typing import Type, TypeVar

from openai_api.logging_config import get_logger
from openai_api.models import (
    SentencesResponse,
    TranslationEvaluationResponse,
    SentenceTranslationsToEvaluate,
    WordCardsResponse,
)

logger = get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
MODEL_NAME = "gpt-4o-2024-08-06"
TARGET_LANGUAGE = "English"
N = 5

load_dotenv()
client = OpenAI()

T = TypeVar("T", bound=BaseModel)


def send_prompt(prompt: str, response_model: Type[T]) -> T:
    logger.info(
        "send_prompt: calling OpenAI model=%s response_model=%s",
        MODEL_NAME,
        response_model.__name__,
    )
    start = time.monotonic()
    try:
        response = client.chat.completions.parse(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_model,
        )
    except OpenAIError as exc:
        logger.error(
            "send_prompt: OpenAI API error model=%s error=%s",
            MODEL_NAME,
            exc,
            exc_info=True,
        )
        raise
    except Exception as exc:
        logger.error(
            "send_prompt: unexpected error model=%s error=%s",
            MODEL_NAME,
            exc,
            exc_info=True,
        )
        raise

    elapsed = time.monotonic() - start
    usage = response.usage
    if usage:
        logger.info(
            "send_prompt: success model=%s latency_s=%.2f "
            "prompt_tokens=%d completion_tokens=%d total_tokens=%d",
            MODEL_NAME,
            elapsed,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )
    else:
        logger.info(
            "send_prompt: success model=%s latency_s=%.2f usage=unavailable",
            MODEL_NAME,
            elapsed,
        )

    return response.choices[0].message.parsed


def get_sentences_for_translation(words: list[str], hsk_level: int | None = None) -> str:
    logger.info(
        "get_sentences_for_translation: words=%s hsk_level=%s",
        words,
        hsk_level,
    )
    prompt_file = PROMPTS_DIR / "get_sentences_for_translation"
    prompt_template = prompt_file.read_text(encoding="utf-8")

    hsk_instruction = (
        f"Keep surrounding vocabulary and grammar complexity appropriate for HSK level {hsk_level}.\n"
        if hsk_level else ""
    )

    prompt = prompt_template.replace("${words}", ", ".join(words))
    prompt = prompt.replace("${n}", str(N))
    prompt = prompt.replace("${targetLanguage}", TARGET_LANGUAGE)
    prompt = prompt.replace("${hskLevelInstruction}", hsk_instruction)

    return send_prompt(prompt, SentencesResponse)


def evaluate_translations(data: SentenceTranslationsToEvaluate) -> str:
    logger.info(
        "evaluate_translations: translation_count=%d",
        len(data.translations),
    )
    prompt_file = PROMPTS_DIR / "evaluate_translations"
    prompt_template = prompt_file.read_text(encoding="utf-8")

    items_text = "\n\n".join(
        f'{i + 1}. Sentence: "{sentence}"\n'
        f'User Translation: "{item.translation}"\n'
        f'Target Word: "{item.word}"'
        for i, item in enumerate(data.translations)
    )

    prompt = prompt_template.replace("${items}", items_text)
    return send_prompt(prompt, TranslationEvaluationResponse)


def get_word_cards(words: list[str], hsk_level: int | None = None) -> WordCardsResponse:
    logger.info(
        "get_word_cards: words=%s hsk_level=%s",
        words,
        hsk_level,
    )
    prompt_file = PROMPTS_DIR / "get_word_cards"
    prompt_template = prompt_file.read_text(encoding="utf-8")

    hsk_instruction = (
        f"The learner is at HSK level {hsk_level}. Keep example sentence vocabulary and grammar appropriate for that level.\n"
        if hsk_level else ""
    )

    prompt = prompt_template.replace("${words}", ", ".join(words))
    prompt = prompt.replace("${hskLevelInstruction}", hsk_instruction)

    return send_prompt(prompt, WordCardsResponse)
