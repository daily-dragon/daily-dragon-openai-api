import pytest
from unittest.mock import MagicMock, patch

from openai_api.auth.cognito import cognito_auth, DailyDragonCognitoToken
from openai_api.openai_api_app import app
from openai_api.models import (
    SentencesResponse,
    SentenceItem,
    WordCardsResponse,
    WordCardItem,
    ExampleSentence,
)
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import openai_api.openai_service as openai_service_module


def dummy_auth():
    return DailyDragonCognitoToken.model_validate({
        "aud": "test-aud",
        "sub": "test-sub",
        "email": "test@example.com",
        "cognito:username": "testuser",
        "email_verified": True,
        "token_use": "id",
    })


@pytest.fixture
def client():
    app.dependency_overrides = dict()
    app.dependency_overrides[cognito_auth.auth_required] = lambda: dummy_auth()
    c = TestClient(app)
    yield c
    app.dependency_overrides = {}


def make_word_cards_response():
    return WordCardsResponse(
        cards=[
            WordCardItem(
                word="\u597d\u53cb",
                pinyin="pengyou",
                meanings=["friend"],
                examples=[
                    ExampleSentence(
                        chinese="\u6211\u6709\u5f89\u597d\u53cb\vo:",
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
    # FastAPI validation rejects empty list with min_length constraint
    # If no constraint is set, the service is called with an empty list;
    # either way the response should not be a 500.
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
