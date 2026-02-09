"""Unit tests for authentication and Cognito configuration."""
from auth.cognito import DailyDragonCognitoToken, DailyDragonCognitoSettings, cognito_auth, settings



def test_daily_dragon_cognito_token_creation():
    """Test creating a DailyDragonCognitoToken."""
    token = DailyDragonCognitoToken(
        aud="test-audience",
        email="user@example.com",
        email_verified=True,
    )
    assert token.aud == "test-audience"
    assert token.email == "user@example.com"
    assert token.email_verified is True


def test_daily_dragon_cognito_token_with_all_fields():
    """Test DailyDragonCognitoToken with all optional fields."""
    token = DailyDragonCognitoToken(
        aud="test-aud",
        auth_time=1234567890,
        **{"cognito:username": "testuser"},
        email="test@example.com",
        email_verified=True,
        event_id="test-event-id",
        exp=1234567900,
        iat=1234567890,
        iss="test-issuer",
        jti="test-jti",
        origin_jti="test-origin-jti",
        sub="test-sub",
        token_use="access",
    )
    assert token.cognito_username == "testuser"
    assert token.auth_time == 1234567890
    assert token.token_use == "access"


def test_daily_dragon_cognito_token_minimal():
    """Test DailyDragonCognitoToken with minimal fields."""
    token = DailyDragonCognitoToken()
    assert token.aud is None
    assert token.email is None
    assert token.email_verified is False


def test_daily_dragon_cognito_settings_defaults():
    """Test DailyDragonCognitoSettings with default values."""
    settings_obj = DailyDragonCognitoSettings()
    assert settings_obj.check_expiration is True
    assert settings_obj.jwt_header_prefix == "Bearer"
    assert settings_obj.jwt_header_name == "Authorization"


def test_daily_dragon_cognito_settings_userpools():
    """Test DailyDragonCognitoSettings has correct userpool configuration."""
    settings_obj = DailyDragonCognitoSettings()
    assert "us" in settings_obj.userpools
    assert settings_obj.userpools["us"]["region"] == "us-west-2"
    assert settings_obj.userpools["us"]["userpool_id"] == "us-west-2_n9Z1AnHRP"
    assert settings_obj.userpools["us"]["app_client_id"] == "6i72m9qe4aj391d195mf7m58rt"


def test_cognito_auth_initialized():
    """Test that cognito_auth is properly initialized."""
    assert cognito_auth is not None


def test_settings_initialized():
    """Test that settings is properly initialized."""
    assert settings is not None
    assert settings.check_expiration is True


def test_daily_dragon_cognito_token_email_verified_default():
    """Test that email_verified defaults to False."""
    token = DailyDragonCognitoToken(email="test@example.com")
    assert token.email_verified is False


def test_daily_dragon_cognito_token_email_verified_true():
    """Test setting email_verified to True."""
    token = DailyDragonCognitoToken(
        email="verified@example.com",
        email_verified=True,
    )
    assert token.email_verified is True


def test_daily_dragon_cognito_token_cognito_username_alias():
    """Test that cognito_username field accepts cognito:username alias."""
    token = DailyDragonCognitoToken.model_validate({
        "cognito:username": "aliasuser"
    })
    assert token.cognito_username == "aliasuser"


def test_daily_dragon_cognito_token_json_serialization():
    """Test that token can be serialized to JSON."""
    token = DailyDragonCognitoToken(
        aud="test-aud",
        email="test@example.com",
        cognito_username="testuser",
    )
    json_data = token.model_dump_json()
    assert "test-aud" in json_data
    assert "test@example.com" in json_data


def test_daily_dragon_cognito_token_from_dict():
    """Test creating token from dictionary."""
    token_dict = {
        "aud": "test-aud",
        "email": "test@example.com",
        "email_verified": True,
    }
    token = DailyDragonCognitoToken.model_validate(token_dict)
    assert token.aud == "test-aud"
    assert token.email == "test@example.com"


def test_settings_jwt_header_configuration():
    """Test JWT header configuration in settings."""
    settings_obj = DailyDragonCognitoSettings()
    assert settings_obj.jwt_header_name == "Authorization"
    assert settings_obj.jwt_header_prefix == "Bearer"


def test_settings_check_expiration_enabled():
    """Test that token expiration check is enabled by default."""
    settings_obj = DailyDragonCognitoSettings()
    assert settings_obj.check_expiration is True


def test_daily_dragon_cognito_token_multiple_instances():
    """Test creating multiple token instances independently."""
    token1 = DailyDragonCognitoToken(
        aud="aud1",
        email="user1@example.com",
    )
    token2 = DailyDragonCognitoToken(
        aud="aud2",
        email="user2@example.com",
    )
    assert token1.aud != token2.aud
    assert token1.email != token2.email


def test_daily_dragon_cognito_token_with_timestamps():
    """Test token with timestamp fields."""
    token = DailyDragonCognitoToken(
        auth_time=1234567890,
        iat=1234567890,
        exp=1234567900,
    )
    assert token.auth_time == 1234567890
    assert token.iat == 1234567890
    assert token.exp == 1234567900


def test_daily_dragon_cognito_token_with_identifiers():
    """Test token with various identifier fields."""
    token = DailyDragonCognitoToken(
        jti="unique-jti",
        origin_jti="origin-jti",
        sub="subject-id",
        event_id="event-id",
    )
    assert token.jti == "unique-jti"
    assert token.origin_jti == "origin-jti"
    assert token.sub == "subject-id"
    assert token.event_id == "event-id"


def test_settings_userpool_region():
    """Test that userpool region is correctly configured."""
    settings_obj = DailyDragonCognitoSettings()
    assert settings_obj.userpools["us"]["region"] == "us-west-2"


def test_settings_userpool_client_id():
    """Test that userpool client ID is correctly configured."""
    settings_obj = DailyDragonCognitoSettings()
    assert settings_obj.userpools["us"]["app_client_id"] == "6i72m9qe4aj391d195mf7m58rt"

