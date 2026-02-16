from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from pydantic import BaseModel

from typing import Type

from openai_api.models import SentencesResponse, TranslationEvaluationResponse, SentenceTranslationsToEvaluate

PROMPTS_DIR = Path(__file__).parent / "prompts"
MODEL_NAME = "gpt-4o-2024-08-06"
TARGET_LANGUAGE = "English"
N = 5

load_dotenv()
client = OpenAI()


def send_prompt(prompt: str, response_model: Type[BaseModel]) -> str:
    response = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format=response_model
    )
    return response.choices[0].message.content


def get_sentences_for_translation(words: list[str]) -> str:
    prompt_file = PROMPTS_DIR / "get_sentences_for_translation"
    prompt_template = prompt_file.read_text(encoding="utf-8")

    prompt = prompt_template.replace("${words}", ", ".join(words))
    prompt = prompt.replace("${n}", str(N))
    prompt = prompt.replace("${targetLanguage}", TARGET_LANGUAGE)

    return send_prompt(prompt, SentencesResponse)


def evaluate_translations(data: SentenceTranslationsToEvaluate) -> str:
    prompt_file = PROMPTS_DIR / "evaluate_translations"
    prompt_template = prompt_file.read_text(encoding="utf-8")

    items_text = "\n".join(
        f'{i + 1}. Sentence: "{item.sentence}"\n'
        f'User Translation: "{item.translation}"\n'
        f'Target Word: "{item.word}"\n'
        for i, item in enumerate(data.translations)
    )

    prompt = f"{prompt_template.rstrip()}{items_text}"

    return send_prompt(prompt, TranslationEvaluationResponse)
