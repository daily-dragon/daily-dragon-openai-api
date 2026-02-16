"""Unit tests for OpenAI service layer."""

import pytest
from unittest.mock import patch, MagicMock, mock_open

from models import SentencesResponse, SentenceItem, TranslationEvaluationResponse, TranslationEvaluationItem, \
    SentenceTranslationsToEvaluate, TranslationItem
from openai_service import send_prompt, get_sentences_for_translation, evaluate_translations


@pytest.fixture
def mock_sentences_response():
    """Fixture for mock sentences response."""
    return SentencesResponse(
        sentences=[
            SentenceItem(word="book", sentence="I read a book every day."),
            SentenceItem(word="pen", sentence="I write with a pen."),
        ]
    )


@pytest.fixture
def mock_evaluation_response():
    """Fixture for mock evaluation response."""
    return TranslationEvaluationResponse(
        evaluations=[
            TranslationEvaluationItem(
                sentence="I read a book.",
                translation="Я читаю книгу.",
                target_word="book",
                word_used="book",
                feedback="Good translation",
                correct_sentence="Я читаю книгу.",
                score=95,
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
    )


@patch("openai_service.client")
def test_send_prompt_success(mock_client, mock_sentences_response):
    """Test send_prompt successfully sends and receives response."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = mock_sentences_response.model_dump_json()
    mock_client.chat.completions.parse.return_value = mock_response

    result = send_prompt("Test prompt", SentencesResponse)

    assert result == mock_sentences_response.model_dump_json()
    mock_client.chat.completions.parse.assert_called_once()


@patch("openai_service.client")
def test_send_prompt_correct_model_used(mock_client):
    """Test that send_prompt uses the correct model."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "response"
    mock_client.chat.completions.parse.return_value = mock_response

    send_prompt("prompt", SentencesResponse)

    call_kwargs = mock_client.chat.completions.parse.call_args[1]
    assert call_kwargs["model"] == "gpt-4o-2024-08-06"


@patch("openai_service.client")
def test_send_prompt_correct_message_format(mock_client):
    """Test that send_prompt sends message in correct format."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "response"
    mock_client.chat.completions.parse.return_value = mock_response

    test_prompt = "This is a test prompt"
    send_prompt(test_prompt, SentencesResponse)

    call_kwargs = mock_client.chat.completions.parse.call_args[1]
    assert call_kwargs["messages"][0]["role"] == "user"
    assert call_kwargs["messages"][0]["content"] == test_prompt


@patch("openai_service.client")
def test_send_prompt_response_format_param(mock_client):
    """Test that send_prompt passes response_format parameter."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "response"
    mock_client.chat.completions.parse.return_value = mock_response

    send_prompt("prompt", SentencesResponse)

    call_kwargs = mock_client.chat.completions.parse.call_args[1]
    assert "response_format" in call_kwargs


@patch("openai_service.client")
@patch("builtins.open", new_callable=mock_open, read_data="Words: ${words}, Count: ${n}, Language: ${targetLanguage}")
def test_get_sentences_for_translation_success(mock_file, mock_client):
    """Test get_sentences_for_translation successfully processes request."""
    mock_response = MagicMock()
    expected_response = SentencesResponse(
        sentences=[
            SentenceItem(word="test", sentence="Test sentence."),
        ]
    )
    mock_response.choices[0].message.content = expected_response.model_dump_json()
    mock_client.chat.completions.parse.return_value = mock_response

    result = get_sentences_for_translation(["book", "pen"])

    assert result == expected_response.model_dump_json()


@patch("openai_service.client")
@patch("builtins.open", new_callable=mock_open, read_data="Words: ${words}, Count: ${n}, Language: ${targetLanguage}")
def test_get_sentences_for_translation_prompt_substitution(mock_file, mock_client):
    """Test that get_sentences_for_translation substitutes template variables."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"sentences": []}'
    mock_client.chat.completions.parse.return_value = mock_response

    get_sentences_for_translation(["book", "pen", "desk"])

    call_kwargs = mock_client.chat.completions.parse.call_args[1]
    prompt = call_kwargs["messages"][0]["content"]

    assert "book, pen, desk" in prompt
    assert "${words}" not in prompt
    assert "5" in prompt
    assert "${n}" not in prompt
    assert "English" in prompt
    assert "${targetLanguage}" not in prompt


@patch("openai_service.client")
@patch("builtins.open", new_callable=mock_open, read_data="Evaluate: ${prompt_template}")
def test_get_sentences_for_translation_single_word(mock_file, mock_client):
    """Test get_sentences_for_translation with single word."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"sentences": []}'
    mock_client.chat.completions.parse.return_value = mock_response

    get_sentences_for_translation(["book"])

    call_kwargs = mock_client.chat.completions.parse.call_args[1]
    prompt = call_kwargs["messages"][0]["content"]

    assert "book" in prompt


@patch("openai_service.client")
@patch("builtins.open", new_callable=mock_open, read_data="Template content")
def test_get_sentences_for_translation_multiple_words(mock_file, mock_client):
    """Test get_sentences_for_translation with multiple words."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"sentences": []}'
    mock_client.chat.completions.parse.return_value = mock_response

    words = ["book", "pen", "desk", "chair", "table"]
    get_sentences_for_translation(words)

    call_kwargs = mock_client.chat.completions.parse.call_args[1]
    prompt = call_kwargs["messages"][0]["content"]

    for word in words:
        assert word in prompt


@patch("openai_service.client")
@patch("builtins.open", new_callable=mock_open, read_data="Evaluate translations:\n")
def test_evaluate_translations_success(mock_file, mock_client, mock_evaluation_response):
    """Test evaluate_translations successfully processes translations."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = mock_evaluation_response.model_dump_json()
    mock_client.chat.completions.parse.return_value = mock_response

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

    result = evaluate_translations(translations)

    assert result == mock_evaluation_response.model_dump_json()


@patch("openai_service.client")
@patch("builtins.open", new_callable=mock_open, read_data="Base prompt:\n")
def test_evaluate_translations_prompt_format(mock_file, mock_client):
    """Test that evaluate_translations formats prompt correctly."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"evaluations": []}'
    mock_client.chat.completions.parse.return_value = mock_response

    translations = SentenceTranslationsToEvaluate(
        translations=[
            TranslationItem(
                word="test",
                sentence="Test sentence.",
                translation="Test translation.",
            ),
        ]
    )

    evaluate_translations(translations)

    call_kwargs = mock_client.chat.completions.parse.call_args[1]
    prompt = call_kwargs["messages"][0]["content"]

    assert "1." in prompt
    assert "Test sentence." in prompt
    assert "Test translation." in prompt
    assert "test" in prompt


@patch("openai_service.client")
@patch("builtins.open", new_callable=mock_open, read_data="Evaluate:\n")
def test_evaluate_translations_multiple_items(mock_file, mock_client):
    """Test evaluate_translations with multiple translation items."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"evaluations": []}'
    mock_client.chat.completions.parse.return_value = mock_response

    translations = SentenceTranslationsToEvaluate(
        translations=[
            TranslationItem(
                word=f"word{i}",
                sentence=f"Sentence {i}.",
                translation=f"Translation {i}.",
            )
            for i in range(3)
        ]
    )

    evaluate_translations(translations)

    call_kwargs = mock_client.chat.completions.parse.call_args[1]
    prompt = call_kwargs["messages"][0]["content"]

    assert "1." in prompt
    assert "2." in prompt
    assert "3." in prompt
    for i in range(3):
        assert f"Sentence {i}." in prompt
        assert f"Translation {i}." in prompt


@patch("openai_service.client")
@patch("builtins.open", new_callable=mock_open, read_data="Prompt template")
def test_get_sentences_for_translation_empty_list(mock_file, mock_client):
    """Test get_sentences_for_translation with empty word list."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"sentences": []}'
    mock_client.chat.completions.parse.return_value = mock_response

    result = get_sentences_for_translation([])

    assert result == '{"sentences": []}'


@patch("openai_service.client")
@patch("builtins.open", new_callable=mock_open, read_data="Template")
def test_evaluate_translations_single_item(mock_file, mock_client):
    """Test evaluate_translations with single translation item."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"evaluations": []}'
    mock_client.chat.completions.parse.return_value = mock_response

    translations = SentenceTranslationsToEvaluate(
        translations=[
            TranslationItem(
                word="single",
                sentence="Single sentence.",
                translation="Single translation.",
            ),
        ]
    )

    evaluate_translations(translations)

    call_kwargs = mock_client.chat.completions.parse.call_args[1]
    prompt = call_kwargs["messages"][0]["content"]

    assert "1." in prompt


@patch("openai_service.client")
@patch("builtins.open", new_callable=mock_open, read_data="Words: ${words}")
def test_get_sentences_for_translation_special_characters(mock_file, mock_client):
    """Test get_sentences_for_translation with special characters."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"sentences": []}'
    mock_client.chat.completions.parse.return_value = mock_response

    get_sentences_for_translation(["café", "résumé", "naïve"])

    call_kwargs = mock_client.chat.completions.parse.call_args[1]
    prompt = call_kwargs["messages"][0]["content"]

    assert "café" in prompt or "caf" in prompt  # Handle encoding


@patch("openai_service.client")
@patch("builtins.open", new_callable=mock_open, read_data="Evaluate")
def test_evaluate_translations_preserves_data(mock_file, mock_client):
    """Test that evaluate_translations preserves translation data in prompt."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"evaluations": []}'
    mock_client.chat.completions.parse.return_value = mock_response

    word = "complex"
    sentence = "The complex problem requires deep analysis."
    translation = "Сложная проблема требует глубокого анализа."

    translations = SentenceTranslationsToEvaluate(
        translations=[
            TranslationItem(
                word=word,
                sentence=sentence,
                translation=translation,
            ),
        ]
    )

    evaluate_translations(translations)

    call_kwargs = mock_client.chat.completions.parse.call_args[1]
    prompt = call_kwargs["messages"][0]["content"]

    assert word in prompt
    assert sentence in prompt
    assert translation in prompt
