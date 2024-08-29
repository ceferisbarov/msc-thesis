from datasets import load_dataset
from transformers import pipeline

ds = load_dataset("rajpurkar/squad")["train"]

# qa_pipeline = pipeline(
#     "question-answering",
#     model="csarron/bert-base-uncased-squad-v1",
#     tokenizer="csarron/bert-base-uncased-squad-v1"
# )

# print(ds["train"][0])

# context = ds["train"][0]["context"]
# question = ds["train"][0]["question"]
# predictions = qa_pipeline({
#     'context': context,
#     'question': question
# })

# print(predictions)

ds = ds.shuffle(seed=42).select(range(200))

with open("squad/long_prompts.txt", "w") as file_:
    for row in ds['question']:
        file_.write(row + "\n")
