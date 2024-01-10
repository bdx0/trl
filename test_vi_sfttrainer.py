import enum
from random import randrange

import pandas as pd
from datasets import load_dataset
from peft import LoraConfig
from transformers import GPT2LMHeadModel, TrainingArguments

from test_util import tokenizer
from trl import SFTTrainer


dataset_name = [
    "imdb",
    "5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca",
    "5CD-AI/Vietnamese-NaturalQA-gg-translated-unrefined",
]

dataset = load_dataset(dataset_name[1], split="train")
eval_dataset = load_dataset(dataset_name[1], split="train[:5%]")
df = pd.DataFrame(dataset)
df.head()

# SFT Train
b = {"human": "USER: ", "gpt": "AI: [/INST]"}
d = {"human": "\n", "gpt": "\n</s>\n<s>[INST]\n"}
system_prompt = "Mày là một AI. Mày sẽ trả lời câu hỏi từ USER một cách đúng đắn và khôn ngoan nhất."


def format_instruction(sample):
    if type(convs) is list:
        # print(convs)
        convs = convs[0]
    convs = convs[:2]
    if False:
        ret = f"<s>[INST]<<SYS>> {system_prompt} <</SYS>>\n"
        for i, c in enumerate(convs):
            if i == len(convs) - 1:
                ret += f"{b[c['from']]}\n{c['value']}"
            else:
                ret += f"{b[c['from']]}\n{c['value']}{d[c['from']]}"
        ret += "\n</s>"
    else:
        ret = ""
        for i, c in enumerate(convs):
            ret += f"{b[c['from']]}: {c['value']}\n"
    return ret


# print(format_instruction(dataset[randrange(len(dataset))]))

model = GPT2LMHeadModel.from_pretrained("gpt2")
# model.resize_token_embeddings(len(tokenizer))

max_seq_length = 512

args = TrainingArguments(
    output_dir="tmp_trainer",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    resume_from_checkpoint=True,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    #    "EleutherAI/gpt-neo-125m",
    eval_dataset=eval_dataset,
    train_dataset=dataset,
    max_seq_length=max_seq_length,
    formatting_func=format_instruction,
    dataset_batch_size=1,
)

# trainer.train()
trainer.train(resume_from_checkpoint=True)
