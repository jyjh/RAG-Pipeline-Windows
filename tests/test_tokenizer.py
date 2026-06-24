from src.utils import get_deepseek_tokenizer


def test_byte_tokenizer_is_reversible():
    tokenizer = get_deepseek_tokenizer("byte")
    text = "entropy = information"

    tokens = tokenizer.encode(text)

    assert tokens
    assert tokenizer.decode(tokens) == text
