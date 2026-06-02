import pytest
from unittest.mock import patch

import openai_api.openai_service as openai_service_module
from openai_api.models import (
    SentencesResponse,
    SentenceItem,
    WordCardsResponse,
    WordCardItem,
    ExampleSentence,
)


def make_word_cards_response():
    return WordCardsResponse(
        cards=[
            WordCardItem(
                word="\u597d\u53cb",
                pinyin="pengyou",
                meanings=["friend"],
                examples=[
                    ExampleSentence(
                        chinese="\u6211\u6709\u5f89\u597d\u53cb",
                        english="I have many friends."
                    )
                ],
            )
        ]
    )


def test_get_word_cards_success(client):
    with patch.object(openai_service_module, "get_word_cards", return_value=make_word_cards_response()):
        response = client.post(
            "/daily-dragon/learning/word-cards",
            json={"words": ["\u597d\u53cb"]},
        )
    assert response.status_code == 200
    data = response.json()
    assert "cards" in data
    assert len(data["cards"]) == 1
    card = data["cards"][0]
    assert card["word"] == "\u597d\u53cb"
    assert card["pinyin"] == "pengyou"
    assert card["meanings"] == ["friend"]
    assert len(card["examples"]) == 1


def test_get_word_cards_with_hsk_level(client):
    with patch.object(openai_service_module, "get_word_cards") as mock_fn:
        mock_fn.return_value = make_word_cards_response()
        client.post(
            "/daily-dragon/learning/word-cards",
            json={"words": ["\u597d\u53cb"], "hsk_level": 2},
        )
        mock_fn.assert_called_once_with(["\u597d\u53cb"], 2)


def test_get_word_cards_without_hsk_level(client):
    with patch.object(openai_service_module, "get_word_cards") as mock_fn:
        mock_fn.return_value = make_word_cards_response()
        client.post(
            "/daily-dragon/learning/word-cards",
            json={"words": ["\u597d\u53cb"]},
        )
        mock_fn.assert_called_once_with(["\u597d\u53cb"], None)


def test_get_word_cards_empty_words(client):
    response = client.post(
        "/daily-dragon/learning/word-cards",
        json={"words": []},
    )
    # FastAPI may reject an empty list or pass it through.
    # Either way it should not crash with a 500.
    assert response.status_code != 500


def test_get_word_cards_service_error(client):
    with patch.object(
        openai_service_module, "get_word_cards", side_effect=Exception("OpenAI failed")
    ):
        response = client.post(
            "/daily-dragon/learning/word-cards",
            json={"words": ["\u597d\u53cb"]},
        )
    assert response.status_code == 500


def test_existing_sentences_endpoint_unaffected(client):
    """Regression guard: the existing practice sentences endpoint still works."""
    mock_result = SentencesResponse(sentences=[SentenceItem(word="\u597d\u53cb", sentence="S1")])
    with patch.object(openai_service_module, "get_sentences_for_translation", return_value=mock_result):
        response = client.post(
            "/daily-dragon/practice/sentences",
            json={"words": ["\u597d\u53cb"]},
        )
    assert response.status_code == 200
    assert "sentences" in response.json()
