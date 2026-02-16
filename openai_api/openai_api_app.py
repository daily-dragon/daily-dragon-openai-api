from fastapi import FastAPI, Depends
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from openai_api import openai_service
from auth.cognito import DailyDragonCognitoToken, cognito_auth

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://d36kc4lmm7sv5n.cloudfront.net",
        "https://daily-dragon.havryliuk.com",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WordsList(BaseModel):
    words: list[str]


@app.post("/daily-dragon/practice/sentences")
def create_practice_sentences(words_list: WordsList,
                              auth: DailyDragonCognitoToken = Depends(cognito_auth.auth_required)):
    return openai_service.get_sentences_for_translation(words_list.words)


class TranslationItem(BaseModel):
    word: str
    sentence: str
    translation: str


class SentenceTranslationsToEvaluate(BaseModel):
    translations: list[TranslationItem]


@app.post("/daily-dragon/practice/evaluate-translations")
def evaluate_translations(translations: SentenceTranslationsToEvaluate,
                          auth: DailyDragonCognitoToken = Depends(cognito_auth.auth_required)):
    return openai_service.evaluate_translations(translations)
