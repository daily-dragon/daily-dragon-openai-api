"""Unit tests for Pydantic models."""

import pytest
from openai_api.models import (
    SentenceItem,
    SentencesResponse,
    TranslationEvaluationItem,
    TranslationEvaluationResponse,
)


def test_sentence_item_creation():
    """Test creating a SentenceItem."""
    item = SentenceItem(word="book", sentence="I read a book yesterday.")
    assert item.word == "book"
    assert item.sentence == "I read a book yesterday."


def test_sentence_item_required_fields():
    """Test SentenceItem validates required fields."""
    with pytest.raises(ValueError):
        SentenceItem(word="book")  # Missing sentence


def test_sentences_response_creation():
    """Test creating a SentencesResponse with multiple items."""
    items = [
        SentenceItem(word="book", sentence="I read a book."),
        SentenceItem(word="pen", sentence="I have a pen."),
    ]
    response = SentencesResponse(sentences=items)
    assert len(response.sentences) == 2
    assert response.sentences[0].word == "book"
    assert response.sentences[1].word == "pen"


def test_sentences_response_empty():
    """Test creating SentencesResponse with empty list."""
    response = SentencesResponse(sentences=[])
    assert len(response.sentences) == 0


def test_translation_evaluation_item_creation():
    """Test creating a TranslationEvaluationItem."""
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


def test_translation_evaluation_item_all_fields():
    """Test that all fields are properly set."""
    item = TranslationEvaluationItem(
        sentence="The cat sat on the mat.",
        translation="猫坐在垫子上。",
        target_word="猫",
        target_word_pinyin="māo",
        word_used="猫",
        feedback="Correct usage of the target word",
        correct_sentence="猫坐在垫子上。",
        score=100,
    )
    assert item.sentence == "The cat sat on the mat."
    assert item.target_word == "猫"
    assert item.feedback == "Correct usage of the target word"


def test_translation_evaluation_response_creation():
    """Test creating TranslationEvaluationResponse."""
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
    assert response.evaluations[1].score == 100


def test_translation_evaluation_response_single_item():
    """Test TranslationEvaluationResponse with single item."""
    item = TranslationEvaluationItem(
        sentence="Test sentence.",
        translation="测试翻译。",
        target_word="测试",
        target_word_pinyin="cèshì",
        word_used="测试",
        feedback="Test feedback",
        correct_sentence="测试翻译。",
        score=85,
    )
    response = TranslationEvaluationResponse(evaluations=[item])
    assert len(response.evaluations) == 1
    assert response.evaluations[0].score == 85


def test_models_json_serialization():
    """Test that models can be serialized to JSON."""
    item = SentenceItem(word="test", sentence="Test sentence.")
    response = SentencesResponse(sentences=[item])
    json_data = response.model_dump_json()
    assert "test" in json_data
    assert "Test sentence" in json_data


def test_models_from_json():
    """Test that models can be deserialized from JSON."""
    json_data = '{"sentences": [{"word": "book", "sentence": "I read a book."}]}'
    response = SentencesResponse.model_validate_json(json_data)
    assert len(response.sentences) == 1
    assert response.sentences[0].word == "book"


def test_translation_evaluation_item_zero_score():
    """Test TranslationEvaluationItem with zero score."""
    item = TranslationEvaluationItem(
        sentence="Wrong translation.",
        translation="错误翻译。",
        target_word="提高",
        target_word_pinyin="tígāo",
        word_used="different",
        feedback="Incorrect word usage",
        correct_sentence="Correct sentence.",
        score=0,
    )
    assert item.score == 0


def test_translation_evaluation_item_perfect_score():
    """Test TranslationEvaluationItem with perfect score."""
    item = TranslationEvaluationItem(
        sentence="Perfect translation.",
        translation="完美翻译。",
        target_word="完美",
        target_word_pinyin="wánměi",
        word_used="完美",
        feedback="Perfect",
        correct_sentence="完美翻译。",
        score=100,
    )
    assert item.score == 100


def test_multiple_sentences_response():
    """Test SentencesResponse with multiple sentences."""
    items = [
        SentenceItem(word=f"word{i}", sentence=f"Sentence {i}.")
        for i in range(5)
    ]
    response = SentencesResponse(sentences=items)
    assert len(response.sentences) == 5
    for i, item in enumerate(response.sentences):
        assert item.word == f"word{i}"


def test_translation_evaluation_response_multiple():
    """Test TranslationEvaluationResponse with multiple evaluations."""
    items = [
        TranslationEvaluationItem(
            sentence=f"Sentence {i}",
            translation=f"Translation {i}",
            target_word=f"word{i}",
            target_word_pinyin=f"pīnyīn{i}",
            word_used=f"word{i}",
            feedback=f"Feedback {i}",
            correct_sentence=f"Correct {i}",
            score=50 + i * 10,
        )
        for i in range(5)
    ]
    response = TranslationEvaluationResponse(evaluations=items)
    assert len(response.evaluations) == 5
    for i, item in enumerate(response.evaluations):
        assert item.score == 50 + i * 10
