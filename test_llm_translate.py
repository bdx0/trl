import pandas as pd
from datasets import load_dataset
from langchain.llms.ollama import Ollama


hh_rlhf = load_dataset("Anthropic/hh-rlhf", split="train")
df = pd.DataFrame(hh_rlhf)
print(df.head())


def item_to_arr(item):
    return [t.strip().replace("\n\n", "\n") for t in item.split("\n\nHuman:") if len(t.strip()) > 0]


s = item_to_arr(df.loc[0]["chosen"])
print(s)


llm = Ollama(
    base_url="http://bobo.home.local:11434",
    model="bdx0/vietcuna",
    # template="{{.Prompt}}",
    # template="<s>[INST]### Human: {{ .Prompt }}[/INST]",
    num_ctx=4000,
    num_gpu=2,
    num_thread=2,
)
prompt = [f"""Hãy dịch câu này sang tiếng Việt: "Human: {t}" """ for t in s]
out = llm.generate(["Xin chào", *prompt])
print(out)
for r in out.generations:
    print(r[0].text)
