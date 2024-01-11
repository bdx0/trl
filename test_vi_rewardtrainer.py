# https://medium.com/towards-generative-ai/reward-model-training-2209d1befb5f
# https://python.plainenglish.io/building-a-reward-model-for-your-llm-using-rlhf-in-python-49abaf4906f
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, pipeline

from test_util import tokenizer
from trl import RewardTrainer
from trl.trainer.training_configs import RewardConfig


reward_config = RewardConfig(
    output_dir="tmp_reward_trainer",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    gradient_accumulation_steps=4,
    resume_from_checkpoint=True,
    evaluation_strategy="no",
    max_length=512,
    remove_unused_columns=False,
    optim="adamw_torch",
    logging_steps=500,
)


# data = data.map(add_margin)
# model_name_or_path = "facebook/opt-350m"
model_name_or_path = "gpt2"
# model_name_or_path = "EleutherAI/gpt-neo-125m"
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1)  # "EleutherAI/gpt-neo-125m",
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model.resize_token_embeddings(len(tokenizer))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

train_dataset = load_dataset("Anthropic/hh-rlhf", split="train[:10%]")
eval_dataset = load_dataset("Anthropic/hh-rlhf", split="train[1%:]")

if False:
    ###########
    # conver string to token with pandas
    ###########
    # không thể dùng cách này, vì token length sẽ không bằng nhau.
    # conver origin dataset to pandas
    df = pd.DataFrame(train_dataset)
    print(df.head())

    # convert function to convert string to token with tokenizer and map data to columns
    # (input_ids_chosen, attention_mask_chosen, input_ids_rejected, attention_mask_rejected)

    def create_input(c, r):
        c1, c2 = tokenizer(c).values()
        r1, r2 = tokenizer(r).values()
        return {
            "input_ids_chosen": c1,
            "attention_mask_chosen": c2,
            "input_ids_rejected": r1,
            "attention_mask_rejected": r2,
        }

    # convert pandas to dataset
    data = Dataset.from_pandas(
        df[["chosen", "rejected"]].apply(lambda x: create_input(*x), axis=1, result_type="expand")
    )
    print(data[:2])

    # pre process data
    ###############
else:
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}
    # Tokenize chosen/rejected pairs of inputs
    # Adapt this section to your needs for custom datasets

    # encode_plus
    # https://stackoverflow.com/questions/61708486/whats-difference-between-tokenizer-encode-and-tokenizer-encode-plus-in-hugging
    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen, **kwargs)
            tokenized_rejected = tokenizer(rejected, **kwargs)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    data = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )


def add_margin(row):
    # Assume you have a score_chosen and score_rejected columns that you want to use to compute the margin
    return {"margin": row["score_chosen"] - row["score_rejected"]}

# data = data.map(add_margin)

# Step 5: Define the Trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=reward_config,
    train_dataset=data,
    eval_dataset=eval_dataset,
)

trainer.train()
# trainer.train(resume_from_checkpoint=True)
text = "Xin chào"
# with torch.no_grad():
#     logits = model(**inputs).logits
# predicted_class_id = logits.argmax().item()
# model.config.id2label[predicted_class_id]

classifier = pipeline("sentiment-analysis", model="./tmp_reward")
classifier(text)