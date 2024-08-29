import pandas as pd
from openai_utils import fetch_openai_response

df = pd.read_parquet("imdb-train.parquet")

print(df.head())

noDF = df[df["label"] == 0].sample(n=100, random_state=123)
yesDF = df[df["label"] == 1].sample(n=100, random_state=123)
df = pd.concat([noDF, yesDF])
df = df.sample(frac=1, random_state=145)

print(df.head())

def ask_gpt(text, use_cache=True):
    messages = [{"role": "system", "content": "You are an assistant that answers all questions with either \"yes\" or \"no\". Do not say any other words. Do not say more than a single word."},
    {"role": "user", "content": f"Does the following query have a positive sentiment?\nQuery: {text}\nAnswer: "}]

    response = fetch_openai_response(messages=messages, use_cache=use_cache)

    return response["choices"][0]["message"]["content"]
