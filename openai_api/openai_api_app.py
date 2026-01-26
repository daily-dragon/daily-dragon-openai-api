import json

from fastapi import FastAPI
from pydantic import BaseModel

import openai_service

app = FastAPI()


class WordsList(BaseModel):
    words: list[str]


@app.post("/daily-dragon/practice/sentences")
def create_practice_sentences(words_list: WordsList):
    sentences = openai_service.get_sentences_for_translation(words_list.words)
    return {
        "sentences": json.dumps(sentences)
    }
