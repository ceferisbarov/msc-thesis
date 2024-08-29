from datetime import timedelta
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

SOURCES = ("imdb/long_prompts.txt",
           "imdb/gpt4o_prompts.txt",
           "imdb/llmlingua_prompts.txt",
           "imdb/llmlingua2_prompts.txt")

def time_string_to_seconds(time_str):
    hours, minutes, seconds_microseconds = time_str.split(':')
    seconds, microseconds = seconds_microseconds.split('.')
    
    time_delta = timedelta(hours=int(hours),
                           minutes=int(minutes),
                           seconds=int(seconds),
                           microseconds=int(microseconds))
    
    return time_delta.total_seconds()

def compare_length(source_path):
    prompts = []
    with open(source_path, "r") as file_:
        for line in file_.readlines():
            prompts.append(line.strip())

    avg_len_long = sum(map(len, prompts)) / len(prompts)
    print("Average length:", avg_len_long)

[compare_length(source) for source in SOURCES]
