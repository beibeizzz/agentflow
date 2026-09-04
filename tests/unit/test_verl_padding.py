from __future__ import annotations


def test_response_lengths_support_dense_prompts_and_nested_responses() -> None:
    import torch
    from tensordict import TensorDict

    from agentflow_rl.verl.padding import response_lengths

    prompts = torch.tensor([[1, 2, 3], [1, 2, 3]])
    responses = torch.nested.nested_tensor_from_jagged(
        torch.tensor([4, 5, 6, 7, 8, 9]),
        offsets=torch.tensor([0, 2, 6]),
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]]
    )
    data = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
        },
        batch_size=2,
    )

    assert response_lengths(data) == ([3, 3], [2, 4])


def test_response_log_probs_are_padded_to_current_mini_batch_width() -> None:
    import torch
    from tensordict import TensorDict

    from agentflow_rl.verl.padding import no_padding_2_padding_compatible

    prompts = torch.tensor([[1, 2, 3], [1, 2, 3]])
    responses = torch.nested.nested_tensor_from_jagged(
        torch.tensor([4, 5, 6, 7, 8, 9]),
        offsets=torch.tensor([0, 2, 6]),
    )
    data = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]]
            ),
        },
        batch_size=2,
    )
    flattened_log_probs = torch.arange(12, dtype=torch.float32)

    result = no_padding_2_padding_compatible(flattened_log_probs, data)

    assert result.tolist() == [[2.0, 3.0, 0.0, 0.0], [7.0, 8.0, 9.0, 10.0]]
