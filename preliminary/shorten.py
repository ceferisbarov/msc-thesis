from tqdm import tqdm

from openai_utils import fetch_openai_response

def shorten(question):
    messages=[
        {"role": "system", "content": "You are a writing assistant. Your task it to shorten the questions provided by the user as much as possible, while preserving the content as much as possible. Do not add any other information to the shortened question. Try to not remove any information as well. Only output the new question. Don't output \"Answer:\". Do not start a newline."},
        {"role": "user", "content": f"Question: {question}\nAnswer: "}
    ]

    response = fetch_openai_response(messages)

    return response["choices"][0]["message"]["content"]

short_prompts = []
with open("long_prompts.txt", "r") as file_:
    for line in tqdm(file_.readlines()):
        response = shorten(line.strip())
        short_prompts.append(response.strip())

with open("short_prompts.txt", "w") as file_:
    for prompt in short_prompts:
        file_.write(prompt + "\n")
