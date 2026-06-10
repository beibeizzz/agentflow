from __future__ import annotations

import warnings
from typing import List

try:
    import vllm.entrypoints.openai.protocol as openai_protocol
    from vllm.entrypoints.openai.protocol import ChatCompletionResponse
except ModuleNotFoundError:
    openai_protocol = None
    ChatCompletionResponse = None

try:
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
except ModuleNotFoundError:
    OpenAIServingChat = None


if ChatCompletionResponse is not None:

    class ChatCompletionResponsePatched(ChatCompletionResponse):
        prompt_token_ids: List[int] | None = None
        response_token_ids: List[int] | None = None

else:
    ChatCompletionResponsePatched = None


original_chat_completion_full_generator = (
    OpenAIServingChat.chat_completion_full_generator if OpenAIServingChat is not None else None
)


async def chat_completion_full_generator(
    self,
    request,
    result_generator,
    request_id: str,
    model_name: str,
    conversation,
    tokenizer,
    request_metadata,
):
    prompt_token_ids: List[int] | None = None
    response_token_ids: List[List[int]] | None = None

    async def _generate_inceptor():
        nonlocal prompt_token_ids, response_token_ids
        async for res in result_generator:
            yield res
            prompt_token_ids = res.prompt_token_ids
            response_token_ids = [output.token_ids for output in res.outputs]

    response = await original_chat_completion_full_generator(
        self,
        request,
        _generate_inceptor(),
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
    )
    response = response.model_copy(
        update={
            "prompt_token_ids": prompt_token_ids,
            "response_token_ids": response_token_ids,
        }
    )

    return response


def instrument_vllm():
    if OpenAIServingChat is None or original_chat_completion_full_generator is None:
        warnings.warn("vllm OpenAIServingChat is unavailable. Skip the instrumentation.")
        return

    if getattr(OpenAIServingChat.chat_completion_full_generator, "_agentflow_patched", False):
        warnings.warn("vllm is already instrumented. Skip the instrumentation.")
        return

    if openai_protocol is not None and ChatCompletionResponsePatched is not None:
        openai_protocol.ChatCompletionResponse = ChatCompletionResponsePatched
    chat_completion_full_generator._agentflow_patched = True
    OpenAIServingChat.chat_completion_full_generator = chat_completion_full_generator


def uninstrument_vllm():
    if OpenAIServingChat is None or original_chat_completion_full_generator is None:
        return
    OpenAIServingChat.chat_completion_full_generator = original_chat_completion_full_generator
