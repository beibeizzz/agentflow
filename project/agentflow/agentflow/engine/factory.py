from typing import Any


def create_llm_engine(
    model_string: str,
    use_cache: bool = False,
    is_multimodal: bool = False,
    **kwargs,
) -> Any:
    print(f"Creating LLM engine for model: {model_string}")

    if any(name in model_string for name in ["gpt", "o1", "o3", "o4"]):
        from .openai import ChatOpenAI

        return ChatOpenAI(
            model_string=model_string,
            use_cache=use_cache,
            is_multimodal=is_multimodal,
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            frequency_penalty=kwargs.get("frequency_penalty", 0.5),
            presence_penalty=kwargs.get("presence_penalty", 0.5),
        )

    if "vllm" in model_string:
        from .vllm import ChatVLLM

        return ChatVLLM(
            model_string=model_string.replace("vllm-", ""),
            base_url=kwargs.get("base_url", "http://localhost:8000/v1"),
            use_cache=use_cache,
            is_multimodal=is_multimodal,
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            frequency_penalty=kwargs.get("frequency_penalty", 1.2),
            max_model_len=kwargs.get("max_model_len", 15200),
            max_seq_len_to_capture=kwargs.get("max_seq_len_to_capture", 15200),
            think_mode=kwargs.get("think_mode", "default"),
        )

    raise ValueError(
        f"Engine {model_string} not supported. "
        "This project keeps only OpenAI-compatible GPT and vLLM engines."
    )
