import time

from fastapi import FastAPI, Depends
from starlette.middleware.cors import CORSMiddleware

from openai_api import openai_service
from openai_api.auth.cognito import DailyDragonCognitoToken, cognito_auth
from openai_api.logging_config import get_logger
from openai_api.models import (
    WordsList,
    DeepPracticeRequest,
    ListeningTextRequest,
    SentencesResponse,
    TranslationEvaluationResponse,
    SentenceTranslationsToEvaluate,
    WordCardsResponse,
    DeepPracticeResponse,
    ListeningTextResponse,
)

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



@app.post("/daily-dragon/practice/sentences", response_model=SentencesResponse)
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


@app.post("/daily-dragon/practice/evaluate-translations", response_model=TranslationEvaluationResponse)
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


@app.post("/daily-dragon/learning/word-cards", response_model=WordCardsResponse)
def get_word_cards(
    words_list: WordsList,
    auth: DailyDragonCognitoToken = Depends(cognito_auth.auth_required),
):
    logger.info(
        "POST /learning/word-cards user=%s word_count=%d hsk_level=%s",
        auth.sub,
        len(words_list.words),
        words_list.hsk_level,
    )
    result = openai_service.get_word_cards(words_list.words, words_list.hsk_level)
    logger.info("POST /learning/word-cards user=%s completed", auth.sub)
    return result


@app.post("/daily-dragon/learning/deep-practice-sentences", response_model=DeepPracticeResponse)
def get_deep_practice_sentences(
    request: DeepPracticeRequest,
    auth: DailyDragonCognitoToken = Depends(cognito_auth.auth_required),
):
    logger.info(
        "POST /learning/deep-practice-sentences user=%s word=%s hsk_level=%s",
        auth.sub,
        request.word,
        request.hsk_level,
    )
    result = openai_service.get_deep_practice_sentences(request.word, request.hsk_level)
    logger.info("POST /learning/deep-practice-sentences user=%s completed", auth.sub)
    return result


@app.post("/daily-dragon/listening/text", response_model=ListeningTextResponse)
def get_listening_text(
    request: ListeningTextRequest,
    auth: DailyDragonCognitoToken = Depends(cognito_auth.auth_required),
):
    logger.info(
        "POST /listening/text user=%s word_count=%d hsk_level=%s size=%s",
        auth.sub,
        len(request.words),
        request.hsk_level,
        request.size,
    )
    result = openai_service.get_listening_text(request.words, request.hsk_level, request.size)
    logger.info("POST /listening/text user=%s completed", auth.sub)
    return result
