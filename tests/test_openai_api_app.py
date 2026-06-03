import pytest
from unittest.mock import patch

import openai_api.openai_service as openai_service_module
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
    mock_result = SentencesResponse(sentences=[SentenceItem(word="\u597d\u53cb", sentence="S1")])
    with patch.object(openai_service_module, "get_sentences_for_translation", return_value=mock_result):
        response = client.post(
            "/daily-dragon/practice/sentences",
            json={"words": ["\u597d\u53cb"]},
        )
    assert response.status_code == 200
    assert "sentences" in response.json()


def test_create_practice_sentences(client):
    mock_result = SentencesResponse(sentences=[SentenceItem(word="书", sentence="我在读一本书。")])
    with patch.object(openai_service_module, "get_sentences_for_translation", return_value=mock_result):
        response = client.post("/daily-dragon/practice/sentences", json={"words": ["书"]})
    assert response.status_code == 200
    data = response.json()
    assert data["sentences"][0]["word"] == "书"


def test_create_practice_sentences_with_hsk(client):
    with patch.object(openai_service_module, "get_sentences_for_translation") as mock_fn:
        mock_fn.return_value = SentencesResponse(sentences=[])
        client.post("/daily-dragon/practice/sentences", json={"words": ["书"], "hsk_level": 2})
        mock_fn.assert_called_once_with(["书"], 2)


def test_evaluate_translations_endpoint(client):
    mock_result = TranslationEvaluationResponse(
        evaluations=[
            TranslationEvaluationItem(
                sentence="I read a book.",
                translation="我读一本书。",
                target_word="书",
                target_word_pinyin="shū",
                word_used="书",
                feedback="Good.",
                correct_sentence="我读一本书。",
                score=9,
            )
        ]
    )
    with patch.object(openai_service_module, "evaluate_translations", return_value=mock_result):
        response = client.post(
            "/daily-dragon/practice/evaluate-translations",
            json={"translations": [{"word": "书", "sentence": "I read a book.", "translation": "我读一本书。"}]},
        )
    assert response.status_code == 200
    assert response.json()["evaluations"][0]["target_word"] == "书"


def test_evaluate_translations_service_error(client):
    with patch.object(openai_service_module, "evaluate_translations", side_effect=Exception("fail")):
        response = client.post(
            "/daily-dragon/practice/evaluate-translations",
            json={"translations": [{"word": "书", "sentence": "s", "translation": "t"}]},
        )
    assert response.status_code == 500
