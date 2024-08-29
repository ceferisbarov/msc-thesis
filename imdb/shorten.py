from llmlingua import PromptCompressor
from tqdm import tqdm

from openai_utils import fetch_openai_response

SOURCE_FILE = "imdb/long_prompts.txt"

# llm_lingua_compressor = PromptCompressor(model_name="NousResearch/Llama-2-7b-hf")

llm_lingua2_compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True, # Whether to use llmlingua-2
    device_map="cpu"
)

def gpt4o(question):
    messages=[
        {"role": "system", "content": "You are a writing assistant. Your task it to shorten the the review provided by the user as much as possible, while preserving the content as much as possible. Do not add any other information to the shortened review. Try to not remove any information as well. Only output the new review. Don't output \"Answer:\". Do not start a newline."},
        {"role": "user", "content": f"Question: {question}\nAnswer: "}
    ]

    response = fetch_openai_response(messages)
    text = response["choices"][0]["message"]["content"]
    text = text.replace("\n", " ")

    return text

# def llmlingua(prompt):
#     compressed_prompt = llm_lingua_compressor.compress_prompt(prompt, instruction="", question="", target_token=200)

#     return compressed_prompt["compressed_prompt"]

def llmlingua2(prompt):
    compressed_prompt = llm_lingua2_compressor.compress_prompt(prompt, rate=0.33, force_tokens = ['\n', '?'])

    return compressed_prompt["compressed_prompt"]

ref_table = {"gpt4o": (gpt4o, "imdb/gpt4o_prompts.txt"),
            #  "llmlingua": (llmlingua, "imdb/llmlingua_prompts.txt"),
             "llmlingua2": (llmlingua2, "imdb/llmlingua2_prompts.txt")}

def shorten(shorten_method, source_file, destination_file):
    short_prompts = []
    with open(source_file, "r") as file_:
        for line in tqdm(file_.readlines()):
            response = shorten_method(line.strip())
            short_prompts.append(response.strip())

    with open(destination_file, "w") as file_:
        for prompt in short_prompts:
            file_.write(prompt + "\n")

for key, value in ref_table.items():
    shorten(value[0], SOURCE_FILE, value[1])
