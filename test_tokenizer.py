#!/usr/bin/env python3
"""Test tokenizer to see how <think> is encoded/decoded"""

import sys
sys.path.insert(0, "/mnt/llm_models/MiniMaxAI_MiniMax-M2-EXL3")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/mnt/llm_models/MiniMaxAI_MiniMax-M2-EXL3")

print("=" * 80)
print("Testing <think> tokenization")
print("=" * 80)

# Test encoding
test_text = "<think>\nThis is my reasoning\n</think>"
print(f"\nTest text: {repr(test_text)}")

tokens = tokenizer.encode(test_text, add_special_tokens=False)
print(f"\nToken IDs: {tokens}")

# Check specific tokens
think_open_id = 200050
think_close_id = 200051

print(f"\n<think> token ID in vocab: {think_open_id}")
print(f"</think> token ID in vocab: {think_close_id}")

# Decode individual tokens
if think_open_id in tokenizer.get_vocab().values():
    decoded_open = tokenizer.decode([think_open_id], skip_special_tokens=False)
    print(f"\nDecode [200050] with skip_special_tokens=False: {repr(decoded_open)}")

    decoded_open_skip = tokenizer.decode([think_open_id], skip_special_tokens=True)
    print(f"Decode [200050] with skip_special_tokens=True: {repr(decoded_open_skip)}")

if think_close_id in tokenizer.get_vocab().values():
    decoded_close = tokenizer.decode([think_close_id], skip_special_tokens=False)
    print(f"\nDecode [200051] with skip_special_tokens=False: {repr(decoded_close)}")

    decoded_close_skip = tokenizer.decode([think_close_id], skip_special_tokens=True)
    print(f"Decode [200051] with skip_special_tokens=True: {repr(decoded_close_skip)}")

# Decode full sequence
print(f"\n\nFull decode with skip_special_tokens=False:")
print(repr(tokenizer.decode(tokens, skip_special_tokens=False)))

print(f"\nFull decode with skip_special_tokens=True:")
print(repr(tokenizer.decode(tokens, skip_special_tokens=True)))

print("\n" + "=" * 80)

# Try the problematic sequence
print("\nTesting problematic generation start sequence:")
gen_prompt_tokens = tokenizer.encode("]~b]ai\n<think>\n", add_special_tokens=False)
print(f"Token IDs for ']~b]ai\\n<think>\\n': {gen_prompt_tokens}")
print(f"Decoded: {repr(tokenizer.decode(gen_prompt_tokens, skip_special_tokens=False))}")
print("=" * 80)
