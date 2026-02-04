import json

from fastapi import FastAPI
from pydantic import BaseModel

import openai_service

app = FastAPI()


class WordsList(BaseModel):
    words: list[str]


@app.post("/daily-dragon/practice/sentences")
def create_practice_sentences(words_list: WordsList):
    return openai_service.get_sentences_for_translation(words_list.words)


class TranslationItem(BaseModel):
    word: str
    sentence: str
    translation: str


class SentenceTranslationsToEvaluate(BaseModel):
    translations: list[TranslationItem]


@app.post("/daily-dragon/practice/evaluate-translations")
def evaluate_translations(translations: SentenceTranslationsToEvaluate):
    return openai_service.evaluate_translations(translations)
