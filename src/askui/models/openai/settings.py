from pydantic import BaseModel, Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAiChatCompletionsCreateSettings(BaseModel):
    """
    Settings for creating chat completions with OpenAI.

    Args:
        top_p (float | None, optional): An alternative to sampling with temperature,
            called nucleus sampling, where the model considers the results of the tokens
            with top_p probability mass. So `0.1` means only the tokens comprising
            the top 10% probability mass are considered. We generally recommend
            altering this or `temperature` but not both.
            Defaults to `None`.

        temperature (float, optional): What sampling temperature to use,
            between `0` and `2`. Higher values like `0.8` will make the output more
            random, while lower values like `0.2` will make it more focused and
            deterministic. We generally recommend altering this or `top_p` but not both.
            Defaults to `0.0`.

        max_tokens (int, optional): The maximum number of tokens that can be generated
            in the chat completion. This value can be used to control costs for text
            generated via API.
            Defaults to `1000`.

        seed (int | None, optional): If specified, the system will make a best effort
            to sample deterministically, such that repeated requests with the same seed
            and parameters should return the same result. Determinism is not guaranteed.
            Defaults to `None`.

        stop (str | list[str] | None, optional): Up to 4 sequences where the API
            will stop generating further tokens. The returned text will not contain the
            stop sequence.
            Defaults to `None`.

        frequency_penalty (float | None, optional): Number between `-2.0` and `2.0`.
            Positive values penalize new tokens based on their existing frequency
            in the text so far, decreasing the model's likelihood to repeat the same
            line verbatim.
            Defaults to `None`.

        presence_penalty (float | None, optional): Number between `-2.0` and `2.0`.
            Positive values penalize new tokens based on whether they appear in the text
            so far, increasing the model's likelihood to talk about new topics.
            Defaults to `None`.
    """

    top_p: float | None = Field(
        default=None,
    )
    temperature: float = Field(
        default=0.0,
    )
    max_tokens: int = Field(
        default=1000,
    )
    seed: int | None = Field(
        default=None,
    )
    stop: str | list[str] | None = Field(
        default=None,
    )
    frequency_penalty: float | None = Field(
        default=None,
    )
    presence_penalty: float | None = Field(
        default=None,
    )


class OpenAiModelSettings(BaseSettings):
    """
    Settings for OpenAI model configuration.

    Args:
        model (str): OpenAI model name. Defaults to "gpt-4o"
        api_key (SecretStr): API key for OpenAI authentication
        base_url (HttpUrl): OpenAI base URL. Defaults to https://api.openai.com/v1
        chat_completions_create_settings (OpenAiChatCompletionsCreateSettings): Settings for ChatCompletions
    """

    model_config = SettingsConfigDict(env_prefix="OPENAI_")
    model: str = Field(default="gpt-4o", description="OpenAI model name")
    api_key: SecretStr | None = Field(
        default=None,
        description="API key for OpenAI authentication",
    )
    base_url: HttpUrl = Field(
        default_factory=lambda: HttpUrl("https://api.openai.com/v1"),
        description="OpenAI base URL",
    )
    chat_completions_create_settings: OpenAiChatCompletionsCreateSettings = Field(
        default_factory=OpenAiChatCompletionsCreateSettings,
        description="Settings for ChatCompletions",
    )
