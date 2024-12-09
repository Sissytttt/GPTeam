from enum import Enum

from dotenv import load_dotenv
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.llms import OpenAI
from langchain.schema import BaseMessage
# from utils.windowai_model import ChatWindowAI

from .cache import chat_json_cache, json_cache
from .model_name import ChatModelName
from .parameters import DEFAULT_FAST_MODEL, DEFAULT_SMART_MODEL
from .spinner import Spinner

load_dotenv()


def get_chat_model(name: ChatModelName, **kwargs) -> BaseChatModel:
    if "model_name" in kwargs:
        del kwargs["model_name"]
    if "model" in kwargs:
        del kwargs["model"]

    if name == ChatModelName.TURBO:
        return ChatOpenAI(model_name=name.value, **kwargs)
    elif name == ChatModelName.GPT4:
        return ChatOpenAI(model_name=name.value, **kwargs)
    elif name == ChatModelName.CLAUDE:
        return ChatAnthropic(model=name.value, **kwargs)
    elif name == ChatModelName.CLAUDE_INSTANT:
        return ChatAnthropic(model=name.value, **kwargs)
    elif name == ChatModelName.WINDOW:
        return ChatWindowAI(model_name=name.value, **kwargs)
    else:
        raise ValueError(f"Invalid model name: {name}")


class ChatModel:
    """Wrapper around the ChatModel class."""
    defaultModel: BaseChatModel
    backupModel: BaseChatModel

    def __init__(
        self,
        default_model_name: ChatModelName = DEFAULT_SMART_MODEL,
        backup_model_name: ChatModelName = DEFAULT_FAST_MODEL,
        **kwargs,
    ):
        self.defaultModel = get_chat_model(default_model_name, **kwargs)
        self.backupModel = get_chat_model(backup_model_name, **kwargs)

    @chat_json_cache(sleep_range=(0, 0))
    async def get_chat_completion(self, messages: list[BaseMessage], **kwargs) -> str:
        try:
            resp = await self.defaultModel.agenerate([messages])
        except Exception:
            resp = await self.backupModel.agenerate([messages])

        return resp.generations[0][0].text

    def get_chat_completion_sync(self, messages: list[BaseMessage], **kwargs) -> str:
        try:
            resp = self.defaultModel.generate([messages])
        except Exception:
            resp = self.backupModel.generate([messages])

        return resp.generations[0][0].text


import langchain
from langchain.chat_models.base import BaseChatModel, SimpleChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from typing import Any, Dict, List, Mapping, Optional, Sequence, TypedDict
import websocket
import uuid
import json
from .general import get_open_port


class MessageDict(TypedDict):
    role: str
    content: str


class RequestDict(TypedDict):
    messages: List[MessageDict]
    temperature: float
    request_id: str


class ResponseDict(TypedDict):
    content: str
    request_id: str


class ChatWindowAI(BaseChatModel):
    model_name: str = "window"
    """Model name to use."""
    temperature: float = 0
    """What sampling temperature to use."""
    streaming: bool = False
    """Whether to stream the results."""
    request_timeout: int = 3600
    """Timeout in seconds for the request."""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "window-chat"

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        output_str = self._call(messages, stop=stop)
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        result = ChatResult(generations=[generation])
        return result

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        return self._generate(messages, stop=stop)

    def _call(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> str:
        request_id = str(uuid.uuid4())
        request: RequestDict = {
            "messages": [],
            "temperature": self.temperature,
            "request_id": request_id,
        }

        for message in messages:
            role = "user"  # default role is user
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"

            request["messages"].append(
                {
                    "role": role,
                    "content": message.content,
                }
            )

        ws = websocket.WebSocket()
        port = get_open_port()
        ws.connect(f"ws://127.0.0.1:{port}/windowmodel")
        ws.send(json.dumps(request))
        message = ws.recv()
        ws.close()

        response: ResponseDict = json.loads(message)

        response_content = response["content"]
        response_request_id = response["request_id"]

        # sanity check that response corresponds to request
        if request_id != response_request_id:
            raise ValueError(
                f"Invalid request ID: {response_request_id}, expected: {request_id}"
            )

        return response_content
