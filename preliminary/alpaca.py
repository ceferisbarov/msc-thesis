from datasets import load_dataset

def is_valid(row):
    if row["input"]:
        return False

    if len(row["instruction"]) < 100:
        return False

    if "\n" in row["instruction"].strip():
        return False

    return True

ds = load_dataset("tatsu-lab/alpaca")["train"]
print(len(ds))

ds = ds.filter(is_valid)
print(len(ds))

ds = ds.shuffle(seed=42).select(range(500))

ds = ds.select_columns(['instruction'])

with open("long_prompts.txt", "w") as file_:
    for row in ds['instruction']:
        file_.write(row + "\n")
