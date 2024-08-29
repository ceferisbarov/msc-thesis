import argparse
from llmlingua import PromptCompressor
from tqdm import tqdm

from openai_utils import fetch_openai_response

parser = argparse.ArgumentParser(description="Get dataset name from CLI")
parser.add_argument("--dataset", choices=["squad", "imdb"], help="Choose either 'squad' or 'imdb'")
args = parser.parse_args()
dataset = args.dataset

SOURCE_FILE = f"{dataset}/long_prompts.txt"

llm_lingua_compressor = PromptCompressor(
                                         model_name="TheBloke/Llama-2-7b-Chat-GPTQ",
                                         device_map="cpu"
                                        )

llm_lingua2_compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True, # Whether to use llmlingua-2
    device_map="cpu"
)

def gpt4o(question):
    messages=[
        {"role": "system", "content": "You are a writing assistant. Your task it to shorten the text provided by the user as much as possible, while preserving the content as much as possible. Do not add any other information to the text. Try not to remove any information as well. Only output the new text. Don't output \"Answer:\". Do not start a newline."},
        {"role": "user", "content": f"Original text: {question}\nShortened text: "}
    ]

    response = fetch_openai_response(messages)
    text = response["choices"][0]["message"]["content"]
    text = text.replace("\n", " ")

    return text

def llmlingua(prompt):
    compressed_prompt = llm_lingua_compressor.compress_prompt(prompt, instruction="", question="", target_token=200)

    return compressed_prompt["compressed_prompt"]

def llmlingua2(prompt):
    compressed_prompt = llm_lingua2_compressor.compress_prompt(prompt, rate=0.33, force_tokens = ['\n', '?'])

    return compressed_prompt["compressed_prompt"]

ref_table = {"gpt4o": (gpt4o,),
             "llmlingua": (llmlingua,),
             "llmlingua2": (llmlingua2,)}

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
    destination_file = f"{dataset}/{key}_prompts.txt"
    shorten(value[0], SOURCE_FILE, destination_file)
