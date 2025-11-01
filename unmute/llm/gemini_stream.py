import os
from typing import Any, AsyncIterator
from google import genai
from unmute.llm.llm_utils import LLMStream

class GeminiStream(LLMStream):
    def __init__(self):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = "models/gemini-2.0-flash-exp"

    async def chat_completion(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        response = await self.client.models.generate_content_stream(
            model=self.model,
            contents=messages,
        )

        async for event in response:
            if event.candidates:
                delta = event.candidates[0].content.parts[0].text
                if delta:
                    yield delta
