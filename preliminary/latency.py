from openai_utils import fetch_openai_response
from datetime import datetime
from tqdm import tqdm

long_prompts = []
with open("long_prompts.txt", "r") as file_:
    for line in file_.readlines():
        long_prompts.append(line.strip())

short_prompts = []
with open("short_prompts.txt", "r") as file_:
    for line in file_.readlines():
        short_prompts.append(line.strip())

for prompt in tqdm(long_prompts):
    messages = [{"role": "user", "content": prompt}]
    start = datetime.now()
    fetch_openai_response(messages, use_cache=False)
    end = datetime.now()
    diff = end - start

    with open("diff_list_long.txt", "a") as file_:
        file_.write("gpt-4o, " + prompt + ", " + str(diff) + "\n")
        
for prompt in tqdm(short_prompts):
    messages = [{"role": "user", "content": prompt}]
    start = datetime.now()
    fetch_openai_response(messages, use_cache=False)
    end = datetime.now()
    diff  = end - start
    with open("diff_list_short.txt", "a") as file_:
        file_.write("gpt-4o, " + prompt + ", " + str(diff) + "\n")

