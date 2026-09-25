from typing import ClassVar

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mule_pattern_learner.configuration import REPOSITORY_ROOT

# Under the repository root, so commands behave the same from any working directory.
ENV_FILE = REPOSITORY_ROOT / ".env"


class Settings(BaseSettings):
    """Connection settings from the repository `.env`; environment variables override it."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = Field(
        default="",
        description="TigerGraph host URL",
    )
    graphname: str = Field(
        default="",
        description="Name of the graph",
    )
    secret: SecretStr = Field(
        default=SecretStr(""),
        description="REST++ secret used to mint auth tokens",
    )

    @field_validator("host", "graphname")
    @classmethod
    def _str_required(cls, v: str, info: ValidationInfo) -> str:
        if not v:
            name = info.field_name or "field"
            raise ValueError(f"{name} must be set in {ENV_FILE} or the environment")
        return v

    @field_validator("secret")
    @classmethod
    def _secret_required(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value():
            raise ValueError(f"secret must be set in {ENV_FILE} or the environment")
        return v
