import pytest
from fastapi.testclient import TestClient

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
def client():
    app.dependency_overrides = dict()
    app.dependency_overrides[cognito_auth.auth_required] = lambda: dummy_auth()
    c = TestClient(app, raise_server_exceptions=False)
    yield c
    app.dependency_overrides = {}
