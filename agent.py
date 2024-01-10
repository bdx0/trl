import fire
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, StoppingCriteria, StoppingCriteriaList


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids: list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        idx = len(self.keywords)
        if np.array_equal(self.keywords, input_ids[0][-idx:]):
            return True
        return False


class Brain:
    def __init__(self) -> None:
        # self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_avaiable() else "cpu"
        self.device = "cpu"
        # model_name_or_path = "gpt2"
        model_name_or_path = "tmp_trainer/checkpoint-26000"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.llm = GPT2LMHeadModel.from_pretrained(model_name_or_path, pad_token_id=self.tokenizer.eos_token_id)
        self.llm.to(self.device)
        self.memory = ""
        stop_words = ["</s>"]
        stop_ids = [self.tokenizer.encode(w) for w in stop_words]
        print(stop_ids)
        stop_criteria = [KeywordsStoppingCriteria(i) for i in stop_ids]
        self.stop_criteria_list = StoppingCriteriaList(stop_criteria)

    def quick_think(self, prompt):
        self.llm.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        input_ids = input_ids.to(self.device)
        outputs = self.llm.generate(
            input_ids=input_ids,
            max_length=1024,
            stopping_criteria=self.stop_criteria_list,
        )
        resp = self.tokenizer.decode(outputs[0])
        # print(resp)
        return resp

    def slow_think(self):
        return


# sft -> Học về cấu trúc của ngôn ngữ, cách sử dụng ngôn ngữ -> Học mặt chữ
# reward -> Học ăn nói sau cho đúng
# rl (reinforcement learning) -> Tư duy sâu về ngôn ngữ
class Agent:
    def __init__(self) -> None:
        self.brain = Brain()

    def chat(self):
        while True:
            prompt = input("> ")
            resp = self.process(prompt)
            print(f"< {resp}")

    def process(self, prompt):
        action = self.brain.quick_think(prompt)

        # action = self.think(prompt)
        # Câu này có nói về mình không?
        # Câu này được bao nhiêu điểm?
        # Câu này có gì hay?
        # Câu này nói gì?
        resp = action
        return resp

    def fast_learn(self):
        # SFT vs less epochs
        # extract reward
        # Fast Learn
        return

    def deep_learn(self):
        # check reward
        # rl from hf
        return

    def think(self, prompt):
        # Mình muốn nó suy nghĩ gì, mình muốn nó suy nghĩ như thế nào? Như thế nào gọi là suy nghĩ? Như thế nào gọi là không suy nghĩ?
        return prompt


if __name__ == "__main__":
    fire.Fire(Agent)
