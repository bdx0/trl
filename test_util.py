import token

from transformers import GPT2Tokenizer


# https://stackoverflow.com/questions/70544129/transformers-asking-to-pad-but-the-tokenizer-does-not-have-a-padding-token
tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2",
    # padding_side="left",
    # truncation_side="left",
    # padding=True,
    # truncation=True,
)
SPECIAL_TOKENS = {
    # chatml
    # <|system|> <|user|> <|assistant|> </s> <s> <|pad|>
    # [ASST] [/ASST] <<SYS>> <</SYS>> [INST] [/INST]
    # "bos_token": "<|im_end|>",
    # "eos_token": "<|im_end|>",
    # "pad_token": "<|pad|>",
    "pad_token": "</s>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "additional_special_tokens": [
        "[USR]",
        "[KG]",
        "[SUB]",
        "[PRED]",
        "[OBJ]",
        "[TRIPLE]",
        "[SEP]",
        "[Q]",
        "[DOM]",
        "[INST]",
        "[/INST]",
        "<<SYS>>",
        "<</SYS>>",
        # "<s>",
        # "</s>",
        "### Instruction:",
        "### Input:",
        "### Output:",
        "### Response:",
    ],
}

# tokenizer.eos_token = "</s>"
# tokenizer.pad_token = "</s>"
# tokenizer.bos_token = "<s>"
# https://stackoverflow.com/questions/70672460/hugging-face-efficient-tokenization-of-unknown-token-in-gpt2
# tokenizer.add_special_tokens(SPECIAL_TOKENS)
tokenizer.pad_token_id = tokenizer.eos_token_id

print(tokenizer)
