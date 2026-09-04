from __future__ import annotations

from typing import Any


def _nested_lengths(tensor: Any) -> list[int]:
    return [int(value) for value in tensor.offsets().diff().tolist()]


def response_lengths(data: Any) -> tuple[list[int], list[int]]:
    """Return concrete prompt and response lengths for mixed TQ batches."""
    prompts = data["prompts"]
    responses = data["responses"]
    attention_mask = data.get("attention_mask")

    if getattr(prompts, "is_nested", False):
        prompt_lengths = _nested_lengths(prompts)
        prompt_width = None
    else:
        prompt_width = int(prompts.shape[1])
        if attention_mask is None:
            prompt_lengths = [prompt_width] * int(prompts.shape[0])
        else:
            prompt_lengths = [
                int(value)
                for value in attention_mask[:, :prompt_width].sum(dim=1).tolist()
            ]

    if getattr(responses, "is_nested", False):
        generated_lengths = _nested_lengths(responses)
    elif attention_mask is not None and prompt_width is not None:
        generated_lengths = [
            int(value)
            for value in attention_mask[:, prompt_width:].sum(dim=1).tolist()
        ]
    else:
        response_width = int(responses.shape[1])
        generated_lengths = [response_width] * int(responses.shape[0])

    if len(prompt_lengths) != len(generated_lengths):
        raise ValueError("prompt and response batches must align")
    if any(length <= 0 for length in prompt_lengths):
        raise ValueError("each training row requires at least one prompt token")
    return prompt_lengths, generated_lengths


def no_padding_2_padding_compatible(tensor: Any, data: Any) -> Any:
    """Slice response logits for dense, nested, and mixed TransferQueue rows."""
    import torch
    import torch.nn.functional as functional

    values = tensor.values() if getattr(tensor, "is_nested", False) else tensor
    prompt_lengths, generated_lengths = response_lengths(data)
    max_response_length = max(generated_lengths)
    sequence_lengths = [
        prompt_length + response_length
        for prompt_length, response_length in zip(
            prompt_lengths, generated_lengths, strict=True
        )
    ]

    sequence_offset = 0
    response_rows = []
    skip_padding = (0, 0) * (values.ndim - 1)
    for sequence_length, response_length in zip(
        sequence_lengths, generated_lengths, strict=True
    ):
        sequence_offset += sequence_length
        response = values[
            sequence_offset - response_length - 1 : sequence_offset - 1
        ]
        response_rows.append(
            functional.pad(
                response,
                (*skip_padding, 0, max_response_length - response_length),
            )
        )

    if sequence_offset != values.shape[0]:
        raise ValueError("flattened model output does not align with sequence lengths")
    return torch.stack(response_rows, dim=0)


__all__ = ["no_padding_2_padding_compatible", "response_lengths"]
