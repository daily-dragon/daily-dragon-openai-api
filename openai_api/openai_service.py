import json

from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from pydantic import BaseModel

PROMPTS_DIR = Path(__file__).parent / "prompts"
MODEL_NAME = "gpt-4o-2024-08-06"
TARGET_LANGUAGE = "English"
N = 5

load_dotenv()
client = OpenAI()


class SentencesResponse(BaseModel):
    sentences: dict[str, str]


def get_sentences_for_translation(words: list[str]) -> dict[str, str]:
    prompt_file = PROMPTS_DIR / "get_sentences_for_translation"
    prompt_template = prompt_file.read_text(encoding="utf-8")

    prompt = prompt_template.replace("${words}", ", ".join(words))
    prompt = prompt.replace("${n}", str(N))
    prompt = prompt.replace("${targetLanguage}", TARGET_LANGUAGE)

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt
    )

    return json.loads(response.output_text)["sentences"]
