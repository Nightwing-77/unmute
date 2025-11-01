import os
import re
import asyncio
from copy import deepcopy
from functools import cache
from typing import Any, AsyncIterator, Protocol, cast

from mistralai import Mistral
from openai import AsyncOpenAI, OpenAI

import google.generativeai as genai

from unmute.kyutai_constants import LLM_SERVER
from ..kyutai_constants import KYUTAI_LLM_API_KEY, KYUTAI_LLM_MODEL

INTERRUPTION_CHAR = "—"
USER_SILENCE_MARKER = "..."


def preprocess_messages_for_llm(chat_history: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []

    for message in chat_history:
        message = deepcopy(message)
        if message["content"].replace(INTERRUPTION_CHAR, "") == "":
            continue
        message["content"] = message["content"].strip().removesuffix(INTERRUPTION_CHAR)

        if output and message["role"] == output[-1]["role"]:
            output[-1]["content"] += " " + message["content"]
        else:
            output.append(message)

    def role_at(index: int) -> str | None:
        if index >= len(output):
            return None
        return output[index]["role"]

    if role_at(0) == "system" and role_at(1) in [None, "assistant"]:
        output = [output[0]] + [{"role": "user", "content": "Hello."}] + output[1:]

    for message in chat_history:
        if (
            message["role"] == "user"
            and message["content"].startswith(USER_SILENCE_MARKER)
            and message["content"] != USER_SILENCE_MARKER
        ):
            message["content"] = message["content"][len(USER_SILENCE_MARKER) :]

    return output


async def rechunk_to_words(iterator: AsyncIterator[str]) -> AsyncIterator[str]:
    buffer = ""
    space_re = re.compile(r"\s+")
    prefix = ""
    async for delta in iterator:
        buffer += delta
        while True:
            match = space_re.search(buffer)
            if match is None:
                break
            chunk = buffer[: match.start()]
            buffer = buffer[match.end() :]
            if chunk != "":
                yield prefix + chunk
            prefix = " "

    if buffer != "":
        yield prefix + buffer


# -----------------------------
# LLM Interface Protocol
# -----------------------------
class LLMStream(Protocol):
    async def chat_completion(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        ...


# -----------------------------
# Mistral
# -----------------------------
class MistralStream:
    def __init__(self):
        self.mistral = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    async def chat_completion(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        event_stream = await self.mistral.chat.stream_async(
            model="mistral-large-latest",
            messages=cast(Any, messages),
            temperature=1.0,
        )
        async for event in event_stream:
            delta = event.data.choices[0].delta.content
            if isinstance(delta, str):
                yield delta


# -----------------------------
# vLLM / OpenAI-compatible server
# -----------------------------
def get_openai_client(
    server_url: str = LLM_SERVER, api_key: str | None = KYUTAI_LLM_API_KEY
) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key or "EMPTY", base_url=server_url + "/v1")


@cache
def autoselect_model() -> str:
    if KYUTAI_LLM_MODEL is not None:
        return KYUTAI_LLM_MODEL
    openai_client = get_openai_client()
    client_sync = OpenAI(
        api_key=openai_client.api_key or "EMPTY",
        base_url=openai_client.base_url,
    )
    models = client_sync.models.list()
    if len(models.data) != 1:
        raise ValueError("Multiple models available; specify one.")
    return models.data[0].id


class VLLMStream:
    def __init__(self, client: AsyncOpenAI, temperature: float = 1.0):
        self.client = client
        self.model = autoselect_model()
        self.temperature = temperature

    async def chat_completion(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=cast(Any, messages),
            stream=True,
            temperature=self.temperature,
        )
        async with stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


# -----------------------------
# Gemini (Google)
# -----------------------------
class GeminiStream:
    def __init__(self, model_name: str = "gemini-1.5-pro", temperature: float = 1.0):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY environment variable.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature

    async def chat_completion(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Stream response text from Gemini model asynchronously."""
        # Convert chat messages to one string prompt (Gemini doesn’t use roles directly)
        prompt = ""
        for m in messages:
            role = m["role"]
            if role == "system":
                prompt += f"[System]: {m['content']}\n"
            elif role == "assistant":
                prompt += f"[Assistant]: {m['content']}\n"
            elif role == "user":
                prompt += f"[User]: {m['content']}\n"

        # Gemini’s SDK is not async, so run it in a thread
        loop = asyncio.get_event_loop()

        def _sync_stream():
            return self.model.generate_content(
                prompt,
                stream=True,
                generation_config={"temperature": self.temperature},
            )

        stream = await loop.run_in_executor(None, _sync_stream)

        for chunk in stream:
            if chunk and hasattr(chunk, "text") and chunk.text:
                yield chunk.text
            await asyncio.sleep(0)  # allow event loop to breathe
