"""
prompt_format.py - Prompt formatting utilities
"""


def format_full_prompt(
    x_prime_text: str,
    prompt_style: str = "oneshot",   # 'oneshot' | 'chatml' | 'raw'
) -> str:
    """
    Format fingerprint prompt with role markers.
    
    - oneshot: "user: {x'}\\nassistant:"
    - chatml (Qwen-like, NO system block):
        "<|im_start|>user\\n{x'}\\n<|im_end|>\\n<|im_start|>assistant\\n"
    - raw: "{x'}"
    """
    if prompt_style == "chatml":
        return (
            "<|im_start|>user\n" + x_prime_text + "\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if prompt_style == "raw":
        return x_prime_text
    # default: oneshot
    return f"user: {x_prime_text}\nassistant:"



