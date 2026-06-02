import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from openai_api.auth.cognito import cognito_auth, DailyDragonCognitoToken
from openai_api.openai_api_app import app


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
def mock_openai_service(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("fastapi.FactoryForBodyParams.get_word_cards", mock, raising=False)
    import openai_api as openai_module
    monkeypatch.setattr(openai_module, "openai_service", mock)
    return mock


@pytest.fixture
def test_client(mock_openai_service):
    app.dependency_overrides = dict()
    app.dependency_overrides[cognito_auth.auth_required] = lambda: dummy_auth()
    client = TestClient(app)
    yield client
    app.dependency_overrides = {}
