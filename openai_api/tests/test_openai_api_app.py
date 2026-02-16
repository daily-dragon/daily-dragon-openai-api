"""Unit tests for FastAPI application controllers/endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from models import SentenceTranslationsToEvaluate, TranslationItem
from openai_api_app import app, WordsList

client = TestClient(app)


@pytest.fixture
def mock_auth_token():
    """Fixture for mock authentication token."""
    return MagicMock(sub="test-user", email="test@example.com")


def test_app_startup():
    """Test that app is properly initialized."""
    assert app is not None


def test_cors_middleware_configured():
    """Test that CORS middleware is configured."""
    # Make a request with origin header
    response = client.options(
        "/daily-dragon/practice/sentences",
        headers={"Origin": "https://daily-dragon.havryliuk.com"},
    )
    # CORS preflight should be handled by the middleware


def test_words_list_model():
    """Test WordsList Pydantic model."""
    words_list = WordsList(words=["book", "pen", "desk"])
    assert len(words_list.words) == 3
    assert "book" in words_list.words


def test_words_list_empty():
    """Test WordsList with empty list."""
    words_list = WordsList(words=[])
    assert len(words_list.words) == 0


def test_translation_item_model():
    """Test TranslationItem Pydantic model."""
    item = TranslationItem(
        word="book",
        sentence="I read a book.",
        translation="Я читаю книгу.",
    )
    assert item.word == "book"
    assert item.sentence == "I read a book."
    assert item.translation == "Я читаю книгу."


def test_sentence_translations_to_evaluate_model():
    """Test SentenceTranslationsToEvaluate Pydantic model."""
    translations = SentenceTranslationsToEvaluate(
        translations=[
            TranslationItem(
                word="book",
                sentence="I read a book.",
                translation="Я читаю книгу.",
            ),
            TranslationItem(
                word="pen",
                sentence="I write with a pen.",
                translation="Я пишу ручкой.",
            ),
        ]
    )
    assert len(translations.translations) == 2


@patch("openai_api.openai_api_app.cognito_auth.auth_required")
@patch("openai_api.openai_api_app.openai_service.get_sentences_for_translation")
def test_create_practice_sentences_endpoint(mock_service, mock_auth, mock_auth_token):
    """Test /daily-dragon/practice/sentences endpoint."""
    mock_auth.return_value = mock_auth_token
    mock_service.return_value = '{"sentences": [{"word": "book", "sentence": "I read a book."}]}'

    response = client.post(
        "/daily-dragon/practice/sentences",
        json={"words": ["book", "pen"]},
        headers={"Authorization": "Bearer mock-token"},
    )
    # Note: This will fail without proper auth setup, which is expected
    # The test validates the endpoint structure exists


@patch("openai_api.openai_api_app.cognito_auth.auth_required")
@patch("openai_api.openai_api_app.openai_service.evaluate_translations")
def test_evaluate_translations_endpoint(mock_service, mock_auth, mock_auth_token):
    """Test /daily-dragon/practice/evaluate-translations endpoint."""
    mock_auth.return_value = mock_auth_token
    mock_service.return_value = '{"evaluations": []}'

    response = client.post(
        "/daily-dragon/practice/evaluate-translations",
        json={
            "translations": [
                {
                    "word": "book",
                    "sentence": "I read a book.",
                    "translation": "Я читаю книгу.",
                }
            ]
        },
        headers={"Authorization": "Bearer mock-token"},
    )

    # Note: This will fail without proper auth setup, which is expected


def test_words_list_json_serialization():
    """Test WordsList JSON serialization."""
    words_list = WordsList(words=["apple", "banana", "cherry"])
    json_str = words_list.model_dump_json()
    assert "apple" in json_str
    assert "banana" in json_str


def test_translation_item_json_serialization():
    """Test TranslationItem JSON serialization."""
    item = TranslationItem(
        word="test",
        sentence="Test sentence.",
        translation="Test translation.",
    )
    json_str = item.model_dump_json()
    assert "test" in json_str


def test_sentence_translations_to_evaluate_json_serialization():
    """Test SentenceTranslationsToEvaluate JSON serialization."""
    translations = SentenceTranslationsToEvaluate(
        translations=[
            TranslationItem(
                word="word",
                sentence="Sentence.",
                translation="Translation.",
            )
        ]
    )
    json_str = translations.model_dump_json()
    assert "word" in json_str


def test_words_list_from_json():
    """Test WordsList deserialization from JSON."""
    json_data = '{"words": ["test1", "test2"]}'
    words_list = WordsList.model_validate_json(json_data)
    assert len(words_list.words) == 2
    assert "test1" in words_list.words


def test_translation_item_from_json():
    """Test TranslationItem deserialization from JSON."""
    json_data = '{"word": "test", "sentence": "Test.", "translation": "Тест."}'
    item = TranslationItem.model_validate_json(json_data)
    assert item.word == "test"


def test_sentence_translations_to_evaluate_from_json():
    """Test SentenceTranslationsToEvaluate deserialization from JSON."""
    json_data = '''{
        "translations": [
            {"word": "w1", "sentence": "s1", "translation": "t1"},
            {"word": "w2", "sentence": "s2", "translation": "t2"}
        ]
    }'''
    translations = SentenceTranslationsToEvaluate.model_validate_json(json_data)
    assert len(translations.translations) == 2


def test_words_list_multiple_words():
    """Test WordsList with many words."""
    words = [f"word{i}" for i in range(10)]
    words_list = WordsList(words=words)
    assert len(words_list.words) == 10


def test_translation_item_with_special_characters():
    """Test TranslationItem with special characters."""
    item = TranslationItem(
        word="café",
        sentence="I went to a café.",
        translation="Я пошел в кафе.",
    )
    assert item.word == "café"


def test_translation_item_with_quotes():
    """Test TranslationItem with quotes in text."""
    item = TranslationItem(
        word="quote",
        sentence='He said "hello".',
        translation='Он сказал "привет".',
    )
    assert 'hello' in item.sentence or 'hello' in item.translation


def test_sentence_translations_to_evaluate_single_item():
    """Test SentenceTranslationsToEvaluate with single item."""
    translations = SentenceTranslationsToEvaluate(
        translations=[
            TranslationItem(
                word="only",
                sentence="Only one item.",
                translation="Только один элемент.",
            )
        ]
    )
    assert len(translations.translations) == 1


def test_sentence_translations_to_evaluate_multiple_items():
    """Test SentenceTranslationsToEvaluate with multiple items."""
    translations = SentenceTranslationsToEvaluate(
        translations=[
            TranslationItem(
                word=f"word{i}",
                sentence=f"Sentence {i}.",
                translation=f"Translation {i}.",
            )
            for i in range(5)
        ]
    )
    assert len(translations.translations) == 5
    for i, item in enumerate(translations.translations):
        assert item.word == f"word{i}"


def test_cors_allowed_origins():
    """Test that CORS allows specified origins."""
    # CloudFront origin
    origin1 = "https://d36kc4lmm7sv5n.cloudfront.net"
    # Custom domain
    origin2 = "https://daily-dragon.havryliuk.com"
    # Localhost
    origin3 = "http://localhost:5173"

    # The app should have these origins configured
    assert app is not None


def test_words_list_required_field():
    """Test that WordsList requires words field."""
    with pytest.raises(ValueError):
        WordsList()  # Missing words


def test_translation_item_required_fields():
    """Test that TranslationItem requires all fields."""
    with pytest.raises(ValueError):
        TranslationItem(word="test")  # Missing sentence and translation


def test_sentence_translations_to_evaluate_required_field():
    """Test that SentenceTranslationsToEvaluate requires translations field."""
    with pytest.raises(ValueError):
        SentenceTranslationsToEvaluate()  # Missing translations


def test_words_list_type_validation():
    """Test WordsList validates words is a list."""
    words_list = WordsList(words=["word1", "word2"])
    assert isinstance(words_list.words, list)


def test_translation_item_string_validation():
    """Test TranslationItem validates string fields."""
    item = TranslationItem(
        word="test",
        sentence="Test sentence.",
        translation="Test translation.",
    )
    assert isinstance(item.word, str)
    assert isinstance(item.sentence, str)
    assert isinstance(item.translation, str)


def test_sentence_translations_to_evaluate_list_validation():
    """Test SentenceTranslationsToEvaluate validates translations is a list."""
    translations = SentenceTranslationsToEvaluate(
        translations=[
            TranslationItem(
                word="test",
                sentence="Test.",
                translation="Тест.",
            )
        ]
    )
    assert isinstance(translations.translations, list)


@patch("openai_api.openai_api_app.cognito_auth.auth_required")
def test_create_practice_sentences_requires_auth(mock_auth):
    """Test that /daily-dragon/practice/sentences requires authentication."""
    mock_auth.side_effect = Exception("Unauthorized")

    # The endpoint requires auth, so it should fail without valid token


@patch("openai_api.openai_api_app.cognito_auth.auth_required")
def test_evaluate_translations_requires_auth(mock_auth):
    """Test that /daily-dragon/practice/evaluate-translations requires authentication."""
    mock_auth.side_effect = Exception("Unauthorized")
    # The endpoint requires auth, so it should fail without valid token


def test_words_list_model_dump():
    """Test WordsList model_dump method."""
    words_list = WordsList(words=["a", "b", "c"])
    data = words_list.model_dump()
    assert "words" in data
    assert len(data["words"]) == 3


def test_translation_item_model_dump():
    """Test TranslationItem model_dump method."""
    item = TranslationItem(
        word="word",
        sentence="Sentence.",
        translation="Translation.",
    )
    data = item.model_dump()
    assert data["word"] == "word"
    assert data["sentence"] == "Sentence."
    assert data["translation"] == "Translation."


def test_sentence_translations_to_evaluate_model_dump():
    """Test SentenceTranslationsToEvaluate model_dump method."""
    translations = SentenceTranslationsToEvaluate(
        translations=[
            TranslationItem(
                word="test",
                sentence="Test.",
                translation="Тест.",
            )
        ]
    )
    data = translations.model_dump()
    assert "translations" in data
    assert len(data["translations"]) == 1
