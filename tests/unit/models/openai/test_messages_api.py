import json
from unittest.mock import MagicMock
import pytest
from askui.models.openai.messages_api import OpenAiMessagesApi
from askui.models.openai.models import OpenAiModel
from askui.models.openai.settings import OpenAiModelSettings
from askui.locators.serializers import VlmLocatorSerializer
from askui.models.shared.agent_message_param import (
    MessageParam,
    TextBlockParam,
    ImageBlockParam,
    Base64ImageSourceParam,
    ToolUseBlockParam,
    ToolResultBlockParam
)
from askui.utils.image_utils import ImageSource
from PIL import Image
import io

@pytest.fixture
def mock_openai_client():
    return MagicMock()

@pytest.fixture
def messages_api(mock_openai_client):
    return OpenAiMessagesApi(client=mock_openai_client)

@pytest.fixture
def model(messages_api):
    settings = OpenAiModelSettings()
    locator_serializer = VlmLocatorSerializer()
    return OpenAiModel(settings=settings, messages_api=messages_api, locator_serializer=locator_serializer)

def test_transform_simple_text_message(messages_api):
    messages = [
        MessageParam(role="user", content="Hello"),
        MessageParam(role="assistant", content="Hi there")
    ]
    transformed = messages_api._transform_messages(messages)
    assert transformed == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ]

def test_transform_multimodal_message(messages_api):
    messages = [
        MessageParam(role="user", content=[
            TextBlockParam(text="What is in this image?"),
            ImageBlockParam(source=Base64ImageSourceParam(data="base64data", media_type="image/png"))
        ])
    ]
    transformed = messages_api._transform_messages(messages)
    assert transformed == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,base64data"}
                }
            ]
        }
    ]

def test_transform_tool_use_message(messages_api):
    messages = [
        MessageParam(role="assistant", content=[
            TextBlockParam(text="Let me check the weather."),
            ToolUseBlockParam(id="call_1", name="get_weather", input={"location": "Berlin"})
        ])
    ]
    transformed = messages_api._transform_messages(messages)
    assert transformed == [
        {
            "role": "assistant",
            "content": "Let me check the weather.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Berlin"}'
                    }
                }
            ]
        }
    ]

def test_transform_tool_result_message(messages_api):
    messages = [
        MessageParam(role="user", content=[
            ToolResultBlockParam(tool_use_id="call_1", content="It's sunny.")
        ])
    ]
    transformed = messages_api._transform_messages(messages)
    assert transformed == [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "It's sunny."
        }
    ]

def test_create_message(messages_api, mock_openai_client):
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "The weather is nice."
    mock_choice.message.tool_calls = None
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_response

    messages = [MessageParam(role="user", content="How is the weather?")]
    response = messages_api.create_message(messages, model="gpt-4o")

    assert response.role == "assistant"
    assert response.content[0].text == "The weather is nice."
    assert response.stop_reason == "end_turn"
    mock_openai_client.chat.completions.create.assert_called_once()

def test_model_get(model, messages_api, mock_openai_client):
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "The image shows a cat."
    mock_choice.message.tool_calls = None
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_response

    img = Image.new('RGB', (100, 100))
    source = ImageSource(img)

    result = model.get("What is in the image?", source, response_schema=None, model="gpt-4o")

    assert result == "The image shows a cat."
    mock_openai_client.chat.completions.create.assert_called_once()

def test_model_locate(model, messages_api, mock_openai_client):
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "<click>500,400</click>"
    mock_choice.message.tool_calls = None
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_response

    # Image resolution is 1000x800
    img = Image.new('RGB', (1000, 800))
    source = ImageSource(img)

    # Internal resolution in OpenAiModel is 1280x800
    # Center offset is (1280 - 1000) // 2 = 140
    # (500 - 140) / 1.0 = 360
    # (400 - 0) / 1.0 = 400

    result = model.locate("button", source, model="gpt-4o")

    assert len(result) == 1
    assert result[0] == (360, 400)
    mock_openai_client.chat.completions.create.assert_called_once()

def test_model_get_with_schema(model, messages_api, mock_openai_client):
    from askui.models.types.response_schemas import ResponseSchemaBase

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"response": {"name": "cat"}}'
    mock_choice.message.tool_calls = None
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_response

    img = Image.new('RGB', (100, 100))
    source = ImageSource(img)

    class CatSchema(ResponseSchemaBase):
        name: str

    result = model.get("What is in the image?", source, response_schema=CatSchema, model="gpt-4o")

    assert result.name == "cat"
    mock_openai_client.chat.completions.create.assert_called_once()
    args, kwargs = mock_openai_client.chat.completions.create.call_args
    assert kwargs["response_format"]["type"] == "json_schema"
