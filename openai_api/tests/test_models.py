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
        translation="Я читаю книгу.",
        target_word="book",
        word_used="book",
        feedback="Good translation",
        correct_sentence="Я читаю книгу.",
        score=95,
    )
    assert item.word_used == "book"
    assert item.score == 95


def test_translation_evaluation_item_all_fields():
    """Test that all fields are properly set."""
    item = TranslationEvaluationItem(
        sentence="The cat sat on the mat.",
        translation="Le chat s'est assis sur le tapis.",
        target_word="cat",
        word_used="cat",
        feedback="Correct usage of the target word",
        correct_sentence="Le chat s'est assis sur le tapis.",
        score=100,
    )
    assert item.sentence == "The cat sat on the mat."
    assert item.target_word == "cat"
    assert item.feedback == "Correct usage of the target word"


def test_translation_evaluation_response_creation():
    """Test creating TranslationEvaluationResponse."""
    items = [
        TranslationEvaluationItem(
            sentence="I read a book.",
            translation="Я читаю книгу.",
            target_word="book",
            word_used="book",
            feedback="Good",
            correct_sentence="Я читаю книгу.",
            score=90,
        ),
        TranslationEvaluationItem(
            sentence="I write with a pen.",
            translation="Я пишу ручкой.",
            target_word="pen",
            word_used="pen",
            feedback="Correct",
            correct_sentence="Я пишу ручкой.",
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
        translation="Test translation.",
        target_word="test",
        word_used="test",
        feedback="Test feedback",
        correct_sentence="Test sentence.",
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
        translation="Неправильный перевод.",
        target_word="word",
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
        translation="Идеальный перевод.",
        target_word="perfect",
        word_used="perfect",
        feedback="Perfect",
        correct_sentence="Идеальный перевод.",
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
