import json
from typing import Any, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI
    from anthropic import Omit
    from anthropic.types import AnthropicBetaParam
    from anthropic.types.beta import (
        BetaThinkingConfigParam,
        BetaToolChoiceParam,
    )

from typing_extensions import override

from askui.models.shared.agent_message_param import (
    ContentBlockParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from askui.models.shared.messages_api import MessagesApi
from askui.models.shared.prompts import SystemPrompt
from askui.models.shared.tools import ToolCollection


class OpenAiMessagesApi(MessagesApi):
    def __init__(
        self,
        client: "OpenAI",
    ) -> None:
        self._client = client

    @override
    def create_message(
        self,
        messages: list[MessageParam],
        model: str,
        tools: "ToolCollection | Omit" = None, # type: ignore
        max_tokens: "int | Omit" = None, # type: ignore
        betas: "list[AnthropicBetaParam] | Omit" = None, # type: ignore
        system: SystemPrompt | None = None,
        thinking: "BetaThinkingConfigParam | Omit" = None, # type: ignore
        tool_choice: "BetaToolChoiceParam | Omit" = None, # type: ignore
        temperature: "float | Omit" = None, # type: ignore
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

        openai_tools = omit
        if not isinstance(_tools, Omit):
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in _tools.to_params()
            ]

        # Map tool_choice
        openai_tool_choice = "auto"
        if not isinstance(_tool_choice, Omit):
            if _tool_choice["type"] == "any":
                openai_tool_choice = "required"
            elif _tool_choice["type"] == "tool":
                openai_tool_choice = {
                    "type": "function",
                    "function": {"name": _tool_choice["name"]},
                }
            elif _tool_choice["type"] == "auto":
                openai_tool_choice = "auto"

        response = self._client.chat.completions.create(
            model=model,
            messages=openai_messages, # type: ignore
            tools=openai_tools if not isinstance(openai_tools, Omit) else None, # type: ignore
            tool_choice=openai_tool_choice if not isinstance(openai_tools, Omit) else None, # type: ignore
            max_tokens=_max_tokens if not isinstance(_max_tokens, Omit) else None,
            temperature=_temperature if not isinstance(_temperature, Omit) else 0.0,
        )

        choice = response.choices[0]
        response_message = choice.message

        content: list[ContentBlockParam] = []
        if response_message.content:
            content.append(TextBlockParam(text=response_message.content))

        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                content.append(
                    ToolUseBlockParam(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=json.loads(tool_call.function.arguments),
                    )
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
            stop_reason=stop_reason_map.get(choice.finish_reason, "end_turn"), # type: ignore
        )

    def _transform_messages(self, messages: list[MessageParam]) -> list[dict[str, Any]]:
        transformed: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "assistant":
                content_text = ""
                tool_calls = []
                if isinstance(msg.content, str):
                    content_text = msg.content
                else:
                    for block in msg.content:
                        if isinstance(block, TextBlockParam):
                            content_text += block.text
                        elif isinstance(block, ToolUseBlockParam):
                            tool_calls.append({
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input)
                                }
                            })

                msg_dict = {"role": "assistant", "content": content_text or None}
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                transformed.append(msg_dict)

            elif msg.role == "user":
                if isinstance(msg.content, str):
                    transformed.append({"role": "user", "content": msg.content})
                else:
                    # Check if it's a tool result message
                    tool_results = [b for b in msg.content if isinstance(b, ToolResultBlockParam)]
                    if tool_results and len(tool_results) == len(msg.content):
                        for result in tool_results:
                            content = ""
                            if isinstance(result.content, str):
                                content = result.content
                            else:
                                for block in result.content:
                                    if isinstance(block, TextBlockParam):
                                        content += block.text

                            transformed.append({
                                "role": "tool",
                                "tool_call_id": result.tool_use_id,
                                "content": content
                            })
                    else:
                        # Regular multimodal user message
                        content_list = []
                        for block in msg.content:
                            if isinstance(block, TextBlockParam):
                                content_list.append({"type": "text", "text": block.text})
                            elif isinstance(block, ImageBlockParam):
                                if hasattr(block.source, "data"): # Base64
                                    content_list.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{block.source.media_type};base64,{block.source.data}"
                                        }
                                    })
                                else: # URL
                                    content_list.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": block.source.url
                                        }
                                    })
                            elif isinstance(block, ToolResultBlockParam):
                                transformed.append({
                                    "role": "tool",
                                    "tool_call_id": block.tool_use_id,
                                    "content": str(block.content)
                                })
                        if content_list:
                            transformed.append({"role": "user", "content": content_list})
        return transformed
