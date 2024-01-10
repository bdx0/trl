import enum
from random import randrange

import pandas as pd
from datasets import load_dataset
from peft import LoraConfig
from transformers import GPT2LMHeadModel, TrainingArguments

from test_util import tokenizer
from trl import SFTTrainer
from trl.trainer.utils import DataCollatorForCompletionOnlyLM


# SFT Train
a = {"human": "USER: ", "gpt": "AI: "}
b = {"human": "USER: ", "gpt": "AI: [/INST]"}
d = {"human": "\n", "gpt": "\n</s>\n<s>[INST]\n"}
system_prompt = "Mày là một AI. Mày sẽ trả lời câu hỏi từ USER một cách đúng đắn và khôn ngoan nhất."


def chat_forinst_one_shot(sample):
    convs = sample["conversations"]
    if len(convs) == 1:
        print(f"convs len: {len(convs)}")
        convs = convs[0]
    return [f"<s>{a[c['from']]} {c['value']}</s>" for _, c in enumerate(convs)]


def chat_forinst_multi_shot(sample):
    convs = sample["conversations"]
    if len(convs) == 1:
        convs = convs[0]
    ret = ""
    for i, c in enumerate(convs):
        ret += f"<s>{a[c['from']]} {c['value']}</s>"
    return [ret]


def chat_forinst_inst_multi_shot(sample):
    convs = sample["conversations"]
    if len(convs) == 1:
        convs = convs[0]
    ret = f"<s>[INST]<<SYS>> {system_prompt} <</SYS>>\n"
    for i, c in enumerate(convs):
        if i == len(convs) - 1:
            ret += f"{b[c['from']]}\n{c['value']}"
        else:
            ret += f"{b[c['from']]}\n{c['value']}{d[c['from']]}"
        ret += "\n</s>"
    return [ret]


def chat_forinst_inst_one_shot(sample):
    convs = sample["conversations"]
    if len(convs) == 1:
        convs = convs[0]
    template = "<s>[INST]<<SYS>> {system_prompt} <</SYS>>\nUSER: {user}\nAI: [/INST]{bot}</s>"
    ret = []
    for i in range(0, len(convs), 2):
        u = convs[i]
        a = convs[i + 1]
        ret.append(template.format_map(
            {"system_prompt": system_prompt, "user": u["value"], "bot": a["value"]}))

    return ret


def load_dataset_with_name(dataset_name=None, format_instruction=None, is_disable=False):
    dataset = load_dataset(dataset_name, split="train")
    eval_dataset = load_dataset(dataset_name, split="train[:5%]")
    df = pd.DataFrame(dataset)
    print(df.head())

    sample = dataset[randrange(len(dataset))]
    print(sample)
    print(format_instruction(sample))
    return dataset, eval_dataset, is_disable


def train_on(dataset_name=None, format_instruction=None, is_disable=False):
    if not dataset_name or len(dataset_name) == 0:
        return
    if not format_instruction:
        return
    if is_disable:
        return
    dataset, eval_dataset, is_disable = load_dataset_with_name(
        dataset_name, format_instruction, is_disable)

    model = GPT2LMHeadModel.from_pretrained(
        "gpt2")  # "EleutherAI/gpt-neo-125m",
    model.resize_token_embeddings(len(tokenizer))

    max_seq_length = 512

    args = TrainingArguments(
        output_dir="tmp_trainer",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        resume_from_checkpoint=True,
    )
    # DataCollatorForCompletionOnlyLM
    # https://huggingface.co/docs/trl/main/en/sft_trainer#advanced-usage

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        eval_dataset=eval_dataset,
        train_dataset=dataset,
        max_seq_length=max_seq_length,
        formatting_func=format_instruction,
        # dataset_batch_size=2,
    )

    # trainer.train()
    trainer.train(resume_from_checkpoint=True)


def imdb_forinst(sample):
    return sample["text"]


datalist = [
    ("imdb", imdb_forinst, True),
    ("5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca", chat_forinst_one_shot, False),
    ("5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca", chat_forinst_multi_shot, False),
    ("5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca", chat_forinst_inst_one_shot, False),
    ("5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca",
     chat_forinst_inst_multi_shot, False),
    ("tsdocode/vi_alpaca_clean", None, False),
    ("uitnlp/vietnamese_students_feedback", None, True),
    ("5CD-AI/Vietnamese-NaturalQA-gg-translated-unrefined", None, True),
]

# load_dataset_with_name(*datalist[0])
for arg in datalist:
    train_on(*arg)
