import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anthropic import Omit
    from anthropic.types import AnthropicBetaParam
    from anthropic.types.beta import (
        BetaThinkingConfigParam,
        BetaToolChoiceParam,
    )
    from openai import OpenAI

from typing_extensions import override

from askui.models.shared.agent_message_param import (
    Base64ImageSourceParam,
    ContentBlockParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from askui.models.shared.messages_api import MessagesApi
from askui.models.shared.prompts import SystemPrompt

if TYPE_CHECKING:
    from askui.models.shared.tools import ToolCollection


class OpenAiMessagesApi(MessagesApi):
    def __init__(
        self,
        client: "OpenAI",
    ) -> None:
        self._client = client

    @property
    def client(self) -> "OpenAI":
        return self._client

    @override
    def create_message(
        self,
        messages: list[MessageParam],
        model: str,
        tools: "ToolCollection | Omit" = None,  # type: ignore
        max_tokens: "int | Omit" = None,  # type: ignore
        betas: "list[AnthropicBetaParam] | Omit" = None,  # type: ignore
        system: SystemPrompt | None = None,
        thinking: "BetaThinkingConfigParam | Omit" = None,  # type: ignore
        tool_choice: "BetaToolChoiceParam | Omit" = None,  # type: ignore
        temperature: "float | Omit" = None,  # type: ignore
    ) -> MessageParam:
        from anthropic import Omit, omit

        # Use provided values or default to omit
        _tools = tools if tools is not None else omit
        _max_tokens = max_tokens if max_tokens is not None else omit
        _betas = betas if betas is not None else omit
        _thinking = thinking if thinking is not None else omit
        _tool_choice = tool_choice if tool_choice is not None else omit
        _temperature = temperature if temperature is not None else omit

        openai_messages = self._transform_messages(messages)

        _system: str | None = None if system is None else str(system)
        if _system:
            openai_messages.insert(0, {"role": "system", "content": _system})

        openai_tools = self._build_openai_tools(_tools)
        openai_tool_choice = self._build_openai_tool_choice(_tool_choice)

        response = self._client.chat.completions.create(
            model=model,
            messages=openai_messages,  # type: ignore[arg-type]
            tools=openai_tools,  # type: ignore[arg-type]
            tool_choice=openai_tool_choice if openai_tools else None,  # type: ignore[arg-type]
            max_tokens=_max_tokens if not isinstance(_max_tokens, Omit) else None,
            temperature=_temperature if not isinstance(_temperature, Omit) else 0.0,
        )

        choice = response.choices[0]
        response_message = choice.message

        content: list[ContentBlockParam] = []
        if response_message.content:
            content.append(TextBlockParam(text=response_message.content))

        if response_message.tool_calls:
            content.extend(
                ToolUseBlockParam(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=json.loads(tool_call.function.arguments),
                )
                for tool_call in response_message.tool_calls
                if tool_call.type == "function"
            )

        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "refusal",
        }

        return MessageParam(
            role="assistant",
            content=content,
            stop_reason=stop_reason_map.get(choice.finish_reason, "end_turn"),
        )

    def _build_openai_tools(
        self, tools: "ToolCollection | Omit"
    ) -> list[dict[str, Any]] | None:
        from anthropic import Omit

        if isinstance(tools, Omit):
            return None

        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],  # type: ignore
                    "description": tool["description"],  # type: ignore
                    "parameters": tool["input_schema"],  # type: ignore
                },
            }
            for tool in tools.to_params()
            if all(key in tool for key in ["name", "description", "input_schema"])
        ]

    def _build_openai_tool_choice(
        self, tool_choice: "BetaToolChoiceParam | Omit"
    ) -> Any:
        from anthropic import Omit

        if isinstance(tool_choice, Omit):
            return "auto"

        if tool_choice["type"] == "any":
            return "required"
        if tool_choice["type"] == "tool":
            return {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
        if tool_choice["type"] == "auto":
            return "auto"
        return "auto"

    def _transform_messages(self, messages: list[MessageParam]) -> list[dict[str, Any]]:
        transformed: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "assistant":
                transformed.append(self._transform_assistant_message(msg))
            elif msg.role == "user":
                transformed.extend(self._transform_user_message(msg))
        return transformed

    def _transform_assistant_message(self, msg: MessageParam) -> dict[str, Any]:
        content_text = ""
        tool_calls = []
        if isinstance(msg.content, str):
            content_text = msg.content
        else:
            for block in msg.content:
                if isinstance(block, TextBlockParam):
                    content_text += block.text
                elif isinstance(block, ToolUseBlockParam):
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            },
                        }
                    )

        msg_dict: dict[str, Any] = {
            "role": "assistant",
            "content": content_text or None,
        }
        if tool_calls:
            msg_dict["tool_calls"] = tool_calls
        return msg_dict

    def _transform_user_message(self, msg: MessageParam) -> list[dict[str, Any]]:
        if isinstance(msg.content, str):
            return [{"role": "user", "content": msg.content}]

        # Check if it's a tool result message
        tool_results = [b for b in msg.content if isinstance(b, ToolResultBlockParam)]
        if tool_results and len(tool_results) == len(msg.content):
            return [self._transform_tool_result(result) for result in tool_results]

        # Regular multimodal user message
        content_list = []
        transformed: list[dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, TextBlockParam):
                content_list.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlockParam):
                content_list.append(self._transform_image_block(block))
            elif isinstance(block, ToolResultBlockParam):
                transformed.append(self._transform_tool_result(block))

        if content_list:
            transformed.insert(0, {"role": "user", "content": content_list})
        return transformed

    def _transform_tool_result(self, result: ToolResultBlockParam) -> dict[str, Any]:
        content = ""
        if isinstance(result.content, str):
            content = result.content
        else:
            for block in result.content:
                if isinstance(block, TextBlockParam):
                    content += block.text
        return {
            "role": "tool",
            "tool_call_id": result.tool_use_id,
            "content": content,
        }

    def _transform_image_block(self, block: ImageBlockParam) -> dict[str, Any]:
        if isinstance(block.source, Base64ImageSourceParam):
            url = f"data:{block.source.media_type};base64,{block.source.data}"
        else:  # URL
            url = block.source.url
        return {
            "type": "image_url",
            "image_url": {"url": url},
        }
