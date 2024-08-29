from datasets import load_dataset
from transformers import pipeline
from evaluate import evaluator

short_sources = (f"SQuAD/gpt4o_prompts.txt", f"SQuAD/llmlingua_prompts.txt", f"SQuAD/llmlingua2_prompts.txt")

task_evaluator = evaluator("question-answering")

qa_pipeline = pipeline(
    "question-answering",
    model="csarron/bert-base-uncased-squad-v1",
    tokenizer="csarron/bert-base-uncased-squad-v1"
)

ds = load_dataset("rajpurkar/squad")["train"]

ds = ds.shuffle(seed=42).select(range(200))

def evaluation(ds, short_source = None):
    if short_source:
        short_questions = []
        with open(short_source, "r") as file_:
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

evaluation(ds)
for short_source in short_sources:
    evaluation(ds, short_source)
