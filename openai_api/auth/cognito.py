from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pydantic.types import Any
from fastapi_cognito import CognitoAuth, CognitoSettings


class DailyDragonCognitoToken(BaseModel):
    aud: Optional[str] = None
    auth_time: Optional[int] = None
    cognito_username: Optional[str] = Field(default=None, validation_alias="cognito:username")
    email: Optional[str] = None
    email_verified: bool = False
    event_id: Optional[str] = None
    exp: Optional[int] = None
    iat: Optional[int] = None
    iss: Optional[str] = None
    jti: Optional[str] = None
    origin_jti: Optional[str] = None
    sub: Optional[str] = None
    token_use: Optional[str] = None


class DailyDragonCognitoSettings(BaseSettings):
    check_expiration: bool = True
    jwt_header_prefix: str = "Bearer"
    jwt_header_name: str = "Authorization"
    userpools: dict[str, dict[str, Any]] = {"us": {
        "region": "us-west-2",
        "userpool_id": "us-west-2_n9Z1AnHRP",
        "app_client_id": "6i72m9qe4aj391d195mf7m58rt"
    }}


settings = DailyDragonCognitoSettings()

cognito_auth = CognitoAuth(
    settings=CognitoSettings.from_global_settings(settings),
    custom_model=DailyDragonCognitoToken
)
