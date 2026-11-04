from typing import Annotated, Literal

from pydantic import BaseModel, Field

ListeningTextSize = Literal["short", "medium", "long"]


class WordsList(BaseModel):
    words: list[str]
    hsk_level: int | None = None


class DeepPracticeRequest(BaseModel):
    word: str
    hsk_level: int | None = None


class ListeningTextRequest(BaseModel):
    words: Annotated[list[str], Field(min_length=1)]
    hsk_level: int | None = None
    size: ListeningTextSize = "medium"


class SentenceItem(BaseModel):
    word: str
    sentence: str


class SentencesResponse(BaseModel):
    sentences: list[SentenceItem]


class TranslationEvaluationItem(BaseModel):
    sentence: str
    translation: str
    target_word: str
    target_word_pinyin: str
    word_used: str
    feedback: str
    correct_sentence: str
    score: int


class TranslationEvaluationResponse(BaseModel):
    evaluations: list[TranslationEvaluationItem]


class TranslationItem(BaseModel):
    word: str
    sentence: str
    translation: str


class SentenceTranslationsToEvaluate(BaseModel):
    translations: list[TranslationItem]


class ExampleSentence(BaseModel):
    chinese: str
    english: str


class WordCardItem(BaseModel):
    word: str
    pinyin: str
    meanings: list[str]
    examples: list[ExampleSentence]


class WordCardsResponse(BaseModel):
    cards: list[WordCardItem]


class DeepPracticeSentenceItem(BaseModel):
    sentence: str
    usage_hint: str


class DeepPracticeResponse(BaseModel):
    sentences: list[DeepPracticeSentenceItem]


class ListeningTextResponse(BaseModel):
    text: str
