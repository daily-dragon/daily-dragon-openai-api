import pytest
from openai_api.models import (
    SentenceItem,
    SentencesResponse,
    TranslationEvaluationItem,
    TranslationEvaluationResponse,
    TranslationItem,
    SentenceTranslationsToEvaluate,
    WordCardsResponse,
    WordCardItem,
    ExampleSentence,
)
from openai_api.openai_api_app import WordsList


def test_sentence_item_creation():
    item = SentenceItem(word="book", sentence="I read a book yesterday.")
    assert item.word == "book"
    assert item.sentence == "I read a book yesterday."


def test_sentence_item_required_fields():
    with pytest.raises(ValueError):
        SentenceItem(word="book")


def test_sentences_response_creation():
    items = [
        SentenceItem(word="book", sentence="I read a book."),
        SentenceItem(word="pen", sentence="I have a pen."),
    ]
    response = SentencesResponse(sentences=items)
    assert len(response.sentences) == 2


def test_sentences_response_empty():
    assert len(SentencesResponse(sentences=[]).sentences) == 0


def test_translation_evaluation_item_creation():
    item = TranslationEvaluationItem(
        sentence="I read a book.",
        translation="我读了一本书。",
        target_word="书",
        target_word_pinyin="shū",
        word_used="书",
        feedback="Good translation",
        correct_sentence="我读了一本书。",
        score=95,
    )
    assert item.word_used == "书"
    assert item.score == 95


def test_translation_evaluation_response_creation():
    items = [
        TranslationEvaluationItem(
            sentence="I read a book.",
            translation="我读了一本书。",
            target_word="书",
            target_word_pinyin="shū",
            word_used="书",
            feedback="Good",
            correct_sentence="我读了一本书。",
            score=90,
        ),
        TranslationEvaluationItem(
            sentence="I write with a pen.",
            translation="我用钢笔写字。",
            target_word="钢笔",
            target_word_pinyin="gāngbǐ",
            word_used="钢笔",
            feedback="Correct",
            correct_sentence="我用钢笔写字。",
            score=100,
        ),
    ]
    response = TranslationEvaluationResponse(evaluations=items)
    assert len(response.evaluations) == 2
    assert response.evaluations[0].score == 90


def test_models_json_serialization():
    response = SentencesResponse(sentences=[SentenceItem(word="test", sentence="Test sentence.")])
    json_data = response.model_dump_json()
    assert "test" in json_data


def test_models_from_json():
    response = SentencesResponse.model_validate_json(
        '{"sentences": [{"word": "book", "sentence": "I read a book."}]}'
    )
    assert response.sentences[0].word == "book"


def test_translation_evaluation_item_zero_score():
    item = TranslationEvaluationItem(
        sentence="Wrong.",
        translation="错误。",
        target_word="提高",
        target_word_pinyin="tígāo",
        word_used="different",
        feedback="Incorrect",
        correct_sentence="Correct.",
        score=0,
    )
    assert item.score == 0


def test_words_list_creation():
    wl = WordsList(words=["book", "pen"])
    assert len(wl.words) == 2
    assert wl.hsk_level is None


def test_words_list_with_hsk_level():
    wl = WordsList(words=["书"], hsk_level=3)
    assert wl.hsk_level == 3


def test_words_list_required():
    with pytest.raises(ValueError):
        WordsList()


def test_translation_item_creation():
    item = TranslationItem(word="book", sentence="I read a book.", translation="Я читаю книгу.")
    assert item.word == "book"


def test_translation_item_required_fields():
    with pytest.raises(ValueError):
        TranslationItem(word="test")


def test_sentence_translations_to_evaluate():
    data = SentenceTranslationsToEvaluate(
        translations=[
            TranslationItem(word="w", sentence="s", translation="t"),
        ]
    )
    assert len(data.translations) == 1


def test_sentence_translations_to_evaluate_required():
    with pytest.raises(ValueError):
        SentenceTranslationsToEvaluate()


def test_word_card_item_creation():
    card = WordCardItem(
        word="朋友",
        pinyin="péngyou",
        meanings=["friend"],
        examples=[ExampleSentence(chinese="我有朋友", english="I have friends.")],
    )
    assert card.word == "朋友"
    assert card.meanings == ["friend"]


def test_word_cards_response_empty():
    assert len(WordCardsResponse(cards=[]).cards) == 0