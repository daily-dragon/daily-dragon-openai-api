import pytest
from unittest.mock import MagicMock, patch, Mock

from openai_api.models import (
    SentencesResponse,
    SentenceItem,
    TranslationEvaluationResponse,
    TranslationEvaluationItem,
    SentenceTranslationsToEvaluate,
    TranslationItem,
    WordCardsResponse,
    WordCardItem,
    ExampleSentence,
)
import openai_api.openai_service as svc


def _mock_parsed_response(parsed_obj):
    """Build a minimal stub of the OpenAI completion response."""
    msg = MagicMock()
    msg.parsed = parsed_obj
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None
    return resp


# ------------------------------------------------
# send_prompt
# ------------------------------------------------

def test_send_prompt_returns_parsed_object():
    expected = SentencesResponse(sentences=[SentenceItem(word="foo", sentence="bar")])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(expected)

    with patch.object(svc, "_get_client", return_value=mock_client):
        result = svc.send_prompt("hello", SentencesResponse)

    assert result is expected
    mock_client.chat.completions.parse.assert_called_once()


def test_send_prompt_with_usage_logging():
    expected = SentencesResponse(sentences=[])
    mock_client = MagicMock()
    resp = _mock_parsed_response(expected)
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_client.chat.completions.parse.return_value = resp

    with patch.object(svc, "_get_client", return_value=mock_client):
        result = svc.send_prompt("hello", SentencesResponse)

    assert result is expected


def test_send_prompt_reraises_openai_error():
    from openai import OpenAIError
    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = OpenAIError("API down")

    with patch.object(svc, "_get_client", return_value=mock_client):
        with pytest.raises(OpenAIError):
            svc.send_prompt("hello", SentencesResponse)


def test_send_prompt_reraises_unexpected_error():
    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = RuntimeError("unexpected")

    with patch.object(svc, "_get_client", return_value=mock_client):
        with pytest.raises(RuntimeError):
            svc.send_prompt("hello", SentencesResponse)


# ------------------------------------------------
# _get_client
# ------------------------------------------------

def test_get_client_returns_same_instance():
    """Lazy client is created once and reused."""
    svc._unitialised_client = None
    with patch("openai_api.openai_service.OpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        c1 = svc._get_client()
        c2 = svc._get_client()
    assert c1 is c2
    mock_cls.assert_called_once()
    svc._unitialised_client = None


# ------------------------------------------------
# get_sentences_for_translation
# ------------------------------------------------

def test_get_sentences_for_translation_calls_send_prompt():
    expected = SentencesResponse(sentences=[])
    with patch.object(svc, "send_prompt", return_value=expected) as mock_send:
        result = svc.get_sentences_for_translation(["word1"], hsk_level=2)
    assert result is expected
    mock_send.assert_called_once()
    prompt_arg, model_arg = mock_send.call_args[0]
    assert "word1" in prompt_arg
    assert "HSK level 2" in prompt_arg
    assert model_arg is SentencesResponse


def test_get_sentences_for_translation_no_hsk_level():
    with patch.object(svc, "send_prompt", return_value=SentencesResponse(sentences=[])) as mock_send:
        svc.get_sentences_for_translation(["word1"])
    prompt_arg = mock_send.call_args[0][0]
    assert "HSK" not in prompt_arg


# ------------------------------------------------
# evaluate_translations
# ------------------------------------------------

def test_evaluate_translations_calls_send_prompt():
    data = SentenceTranslationsToEvaluate(
        translations=[TranslationItem(word="foo", sentence="S1", translation="T1")]
    )
    expected = TranslationEvaluationResponse(evaluations=[])
    with patch.object(svc, "send_prompt", return_value=expected) as mock_send:
        result = svc.evaluate_translations(data)
    assert result is expected
    mock_send.assert_called_once()
    prompt_arg, model_arg = mock_send.call_args[0]
    assert "S1" in prompt_arg
    assert "T1" in prompt_arg
    assert model_arg is TranslationEvaluationResponse


# ------------------------------------------------
# get_word_cards
# ------------------------------------------------

def test_get_word_cards_calls_send_prompt():
    expected = WordCardsResponse(cards=[])
    with patch.object(svc, "send_prompt", return_value=expected) as mock_send:
        result = svc.get_word_cards(["\u597d"], hsk_level=1)
    assert result is expected
    mock_send.assert_called_once()
    prompt_arg, model_arg = mock_send.call_args[0]
    assert "\u597d" in prompt_arg
    assert "HSK level 1" in prompt_arg
    assert model_arg is WordCardsResponse


def test_get_word_cards_no_hsk_level():
    with patch.object(svc, "send_prompt", return_value=WordCardsResponse(cards=[])) as mock_send:
        svc.get_word_cards(["\u597d"])
    prompt_arg = mock_send.call_args[0][0]
    assert "HSK" not in prompt_arg


def test_get_word_cards_passes_correct_model():
    with patch.object(svc, "send_prompt", return_value=WordCardsResponse(cards=[])) as mock_send:
        svc.get_word_cards(["\u597d"])
    _, model_arg = mock_send.call_args[0]
    assert model_arg is WordCardsResponse
