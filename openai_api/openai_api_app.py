import time

from fastapi import FastAPI, Depends
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from openai_api import openai_service
from openai_api.auth.cognito import DailyDragonCognitoToken, cognito_auth
from openai_api.logging_config import get_logger
from openai_api.models import SentencesResponse, TranslationEvaluationResponse, SentenceTranslationsToEvaluate

logger = get_logger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://d36kc4lmm7sv5n.cloudfront.net",
        "https://daily-dragon.havryliuk.com",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WordsList(BaseModel):
    words: list[str]
    hsk_level: int | None = None


@app.post("/daily-dragon/practice/sentences")
def create_practice_sentences(
    words_list: WordsList,
    auth: DailyDragonCognitoToken = Depends(cognito_auth.auth_required),
):
    logger.info(
        "POST /practice/sentences user=%s word_count=%d hsk_level=%s",
        auth.sub,
        len(words_list.words),
        words_list.hsk_level,
    )
    result = openai_service.get_sentences_for_translation(words_list.words, words_list.hsk_level)
    logger.info("POST /practice/sentences user=%s completed", auth.sub)
    return result


@app.post("/daily-dragon/practice/evaluate-translations")
def evaluate_translations(
    translations: SentenceTranslationsToEvaluate,
    auth: DailyDragonCognitoToken = Depends(cognito_auth.auth_required),
):
    logger.info(
        "POST /practice/evaluate-translations user=%s translation_count=%d",
        auth.sub,
        len(translations.translations),
    )
    result = openai_service.evaluate_translations(translations)
    logger.info("POST /practice/evaluate-translations user=%s completed", auth.sub)
    return result
