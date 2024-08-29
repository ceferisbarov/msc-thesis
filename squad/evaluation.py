from datasets import load_dataset
from transformers import pipeline
from evaluate import evaluator

task_evaluator = evaluator("question-answering")

qa_pipeline = pipeline(
    "question-answering",
    model="csarron/bert-base-uncased-squad-v1",
    tokenizer="csarron/bert-base-uncased-squad-v1"
)

ds = load_dataset("rajpurkar/squad")["train"]

ds = ds.shuffle(seed=42).select(range(200))

results = task_evaluator.compute(
    model_or_pipeline=qa_pipeline,
    data=ds,
    metric="squad",
)
print(results)

short_questions = []
with open("short_prompts.txt", "r") as file_:
    for line in file_.readlines():
        short_questions.append(line.strip())

ds = ds.add_column("short_question", short_questions)
ds = ds.remove_columns("question")
ds = ds.rename_column("short_question", "question")

results = task_evaluator.compute(
    model_or_pipeline=qa_pipeline,
    data=ds,
    metric="squad",
)
print(results)