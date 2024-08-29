from tqdm import tqdm
from imdb.imdb import ask_gpt, df
from sklearn.metrics import confusion_matrix, f1_score

short_sources = ("imdb/gpt4o_prompts.txt", "imdb/llmlingua_prompts.txt", "imdb/llmlingua2_prompts.txt")

def is_positive(text):
    if "yes" in text.lower():
        return 1

    return 0

def evaluation(df, short_source = None):
    if short_source:
        short_reviews = []
        with open(short_source, "r") as file_:
            for line in file_.readlines():
                short_reviews.append(line.strip())

        df["text"] = short_reviews

    ground_labels = []
    predicted_labels = []
    for _, row in tqdm(df.iterrows()):
        response = ask_gpt(row["text"], use_cache=True)
        label = is_positive(response)

        predicted_labels.append(label)
        ground_labels.append(row["label"])
        
    print("****************************************")
    print(confusion_matrix(ground_labels, predicted_labels))
    print(f1_score(ground_labels, predicted_labels))
    print()

evaluation(df)
for short_source in short_sources:
    evaluation(df, short_source)
