from pydantic import BaseModel


class SentenceItem(BaseModel):
    word: str
    sentence: str


class SentencesResponse(BaseModel):
    sentences: list[SentenceItem]


class TranslationEvaluationItem(BaseModel):
    sentence: str
    translation: str
    target_word: str
    word_used: str
    feedback: str
    correct_sentence: str
    score: int

class TranslationEvaluationResponse(BaseModel):
    evaluations: list[TranslationEvaluationItem]