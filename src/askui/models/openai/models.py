import json
import logging
from typing import Type, cast

from typing_extensions import override

from askui.locators.locators import Locator
from askui.locators.serializers import VlmLocatorSerializer
from askui.models.anthropic.models import build_system_prompt_locate
from askui.models.anthropic.utils import extract_click_coordinates
from askui.models.exceptions import (
    ElementNotFoundError,
    QueryNoResponseError,
    QueryUnexpectedResponseError,
)
from askui.models.models import GetModel, LocateModel, ModelComposition
from askui.models.shared.agent_message_param import (
    Base64ImageSourceParam,
    ContentBlockParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
)
from askui.models.shared.prompts import SystemPrompt
from askui.models.types.geometry import PointList
from askui.models.types.response_schemas import ResponseSchema, to_response_schema
from askui.prompts.get_prompts import SYSTEM_PROMPT_GET
from askui.utils.excel_utils import OfficeDocumentSource
from askui.utils.image_utils import (
    ImageSource,
    image_to_base64,
    scale_coordinates,
    scale_image_to_fit,
)
from askui.utils.pdf_utils import PdfSource
from askui.utils.source_utils import Source

from .messages_api import OpenAiMessagesApi
from .settings import OpenAiModelSettings

logger = logging.getLogger(__name__)


class OpenAiModel(GetModel, LocateModel):
    def __init__(
        self,
        settings: OpenAiModelSettings,
        messages_api: OpenAiMessagesApi,
        locator_serializer: VlmLocatorSerializer,
    ) -> None:
        self._settings = settings
        self._messages_api = messages_api
        self._locator_serializer = locator_serializer

    def _inference(
        self,
        image: ImageSource,
        prompt: str,
        system: SystemPrompt,
        model: str,
        response_schema: Type[ResponseSchema] | None = None,
    ) -> ResponseSchema | str:
        resolution = (1280, 800)

        scaled_image = scale_image_to_fit(
            image.root,
            resolution,
        )

        if response_schema is not None:
            return self._inference_with_schema(
                image_data=image_to_base64(scaled_image),
                media_type="image/png",
                prompt=prompt,
                system=system,
                model=model,
                response_schema=response_schema,
            )

        message = self._messages_api.create_message(
            messages=[
                MessageParam(
                    role="user",
                    content=cast(
                        "list[ContentBlockParam]",
                        [
                            ImageBlockParam(
                                source=Base64ImageSourceParam(
                                    data=image_to_base64(scaled_image),
                                    media_type="image/png",
                                ),
                            ),
                            TextBlockParam(
                                text=prompt,
                            ),
                        ],
                    ),
                )
            ],
            model=model,
            system=system,
        )
        content: list[ContentBlockParam] = (
            message.content
            if isinstance(message.content, list)
            else [TextBlockParam(text=message.content)]
        )
        if len(content) == 0 or not isinstance(content[0], TextBlockParam):
            error_msg = "Unexpected response from OpenAI API: No text content"
            raise QueryNoResponseError(error_msg, prompt)

        return content[0].text

    def _inference_with_schema(
        self,
        image_data: str,
        media_type: str,
        prompt: str,
        system: SystemPrompt,
        model: str,
        response_schema: Type[ResponseSchema],
    ) -> ResponseSchema:
        _response_schema = to_response_schema(response_schema)

        schema = _response_schema.model_json_schema()
        # Clean schema refs if needed, similar to OpenRouterModel
        from askui.models.openrouter.model import _clean_schema_refs

        _clean_schema_refs(schema)

        defs = schema.pop("$defs", None)
        schema_response_wrapper = {
            "type": "object",
            "properties": {"response": schema},
            "additionalProperties": False,
            "required": ["response"],
        }
        if defs:
            schema_response_wrapper["$defs"] = defs

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response_schema",
                "schema": schema_response_wrapper,
                "strict": True,
            },
        }

        openai_messages = [
            {"role": "system", "content": str(system)},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image_data}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        response = self._messages_api.client.chat.completions.create(  # type: ignore[call-overload]
            model=model,
            messages=openai_messages,
            response_format=response_format,
        )

        model_response = response.choices[0].message.content
        if model_response is None:
            msg = "No response from OpenAI"
            raise QueryNoResponseError(msg, prompt)

        try:
            response_json = json.loads(model_response)
            validated_response = _response_schema.model_validate(
                response_json["response"]
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise QueryUnexpectedResponseError(
                str(e), prompt, [TextBlockParam(text=model_response)]
            ) from e
        else:
            return validated_response.root

    @override
    def locate(
        self,
        locator: str | Locator,
        image: ImageSource,
        model: ModelComposition | str,
    ) -> PointList:
        if not isinstance(model, str):
            error_msg = "Model composition is not supported for OpenAI"
            raise NotImplementedError(error_msg)
        locator_serialized = (
            self._locator_serializer.serialize(locator)
            if isinstance(locator, Locator)
            else locator
        )
        try:
            prompt = f"Click on {locator_serialized}"
            resolution = (1280, 800)
            screen_width = resolution[0]
            screen_height = resolution[1]
            content = self._inference(
                image=image,
                prompt=prompt,
                system=build_system_prompt_locate(
                    str(screen_width), str(screen_height)
                ),
                model=model,
            )
            return [
                scale_coordinates(
                    extract_click_coordinates(content),
                    image.root.size,
                    resolution,
                    inverse=True,
                )
            ]
        except Exception as e:
            raise ElementNotFoundError(locator, locator_serialized) from e

    @override
    def get(
        self,
        query: str,
        source: Source,
        response_schema: Type[ResponseSchema] | None,
        model: str,
    ) -> ResponseSchema | str:
        if isinstance(source, (PdfSource, OfficeDocumentSource)):
            err_msg = (
                f"PDF or Office Document processing is not supported for the model: "
                f"{model}"
            )
            raise NotImplementedError(err_msg)
        try:
            return self._inference(
                image=source,
                prompt=query,
                system=SYSTEM_PROMPT_GET,
                model=model,
                response_schema=response_schema,
            )
        except Exception as e:
            if isinstance(e, QueryNoResponseError):
                raise
            raise QueryUnexpectedResponseError(str(e), query, []) from e
