from openai_api.auth.cognito import DailyDragonCognitoToken, DailyDragonCognitoSettings, cognito_auth, settings


def test_daily_dragon_cognito_token_creation():
    token = DailyDragonCognitoToken(aud="test-audience", email="user@example.com", email_verified=True)
    assert token.aud == "test-audience"
    assert token.email == "user@example.com"
    assert token.email_verified is True


def test_daily_dragon_cognito_token_with_all_fields():
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
    token = DailyDragonCognitoToken()
    assert token.aud is None
    assert token.email is None
    assert token.email_verified is False


def test_daily_dragon_cognito_settings_defaults():
    s = DailyDragonCognitoSettings()
    assert s.check_expiration is True
    assert s.jwt_header_prefix == "Bearer"
    assert s.jwt_header_name == "Authorization"


def test_daily_dragon_cognito_settings_userpools():
    s = DailyDragonCognitoSettings()
    assert "us" in s.userpools
    assert s.userpools["us"]["region"] == "us-west-2"
    assert s.userpools["us"]["userpool_id"] == "us-west-2_n9Z1AnHRP"
    assert s.userpools["us"]["app_client_id"] == "6i72m9qe4aj391d195mf7m58rt"


def test_cognito_auth_initialized():
    assert cognito_auth is not None


def test_settings_initialized():
    assert settings is not None
    assert settings.check_expiration is True


def test_daily_dragon_cognito_token_email_verified_default():
    assert DailyDragonCognitoToken(email="test@example.com").email_verified is False


def test_daily_dragon_cognito_token_cognito_username_alias():
    token = DailyDragonCognitoToken.model_validate({"cognito:username": "aliasuser"})
    assert token.cognito_username == "aliasuser"


def test_daily_dragon_cognito_token_from_dict():
    token = DailyDragonCognitoToken.model_validate(
        {"aud": "test-aud", "email": "test@example.com", "email_verified": True}
    )
    assert token.aud == "test-aud"


def test_daily_dragon_cognito_token_with_timestamps():
    token = DailyDragonCognitoToken(auth_time=1234567890, iat=1234567890, exp=1234567900)
    assert token.auth_time == 1234567890
    assert token.exp == 1234567900


def test_daily_dragon_cognito_token_with_identifiers():
    token = DailyDragonCognitoToken(jti="unique-jti", origin_jti="origin-jti", sub="subject-id")
    assert token.jti == "unique-jti"
    assert token.sub == "subject-id"


def test_daily_dragon_cognito_token_multiple_instances():
    t1 = DailyDragonCognitoToken(aud="aud1", email="user1@example.com")
    t2 = DailyDragonCognitoToken(aud="aud2", email="user2@example.com")
    assert t1.aud != t2.aud