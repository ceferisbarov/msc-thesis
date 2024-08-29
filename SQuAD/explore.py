from datetime import timedelta
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def time_string_to_seconds(time_str):
    # Split the time string into its components
    hours, minutes, seconds_microseconds = time_str.split(':')
    seconds, microseconds = seconds_microseconds.split('.')
    
    # Convert to a timedelta object
    time_delta = timedelta(hours=int(hours),
                           minutes=int(minutes),
                           seconds=int(seconds),
                           microseconds=int(microseconds))
    
    # Return the total number of seconds
    return time_delta.total_seconds()

SOURCES = ("SQuAD/long_prompts.txt",
           "SQuAD/gpt4o_prompts.txt",
           "SQuAD/llmlingua_prompts.txt",
           "SQuAD/llmlingua2_prompts.txt")

def compare_length(source_path):
    prompts = []
    with open(source_path, "r") as file_:
        for line in file_.readlines():
            prompts.append(line.strip())

    avg_len_long = sum(map(len, prompts)) / len(prompts)
    print("Average length:", avg_len_long)

[compare_length(source) for source in SOURCES]

def plot_length_vs_latency(model_name="gpt-4o"):
    prompts = []
    latencies = []
    with open("diff_list_long.txt", "r") as long_file, open("diff_list_short.txt", "r") as short_file:
        for line in long_file.readlines() + short_file.readlines():
            if not line.strip():
                continue
            
            model, prompt, latency = line.split("~")
            if model == model_name:
                prompts.append(prompt)
                latencies.append(time_string_to_seconds(latency))

    prompt_lengths = [len(i) for i in prompts]

    data = pd.DataFrame({
        'Prompt Length': prompt_lengths,
        'Latency': latencies
    })

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Prompt Length', y='Latency', s=100, color='blue')

    # Adding labels and title
    plt.title('Prompt Length vs. Latency')
    plt.xlabel('Prompt Length (number of characters)')
    plt.ylabel('Latency (seconds)')

    # Display the plot
    plt.grid(True)
    plt.savefig(f'{model_name}_length_vs_latency.png', dpi=300, bbox_inches='tight')
    plt.show()

    correlation, _ = pearsonr(prompt_lengths, latencies)
    print(f'Pearson correlation: {correlation}')

    correlation, p_value = spearmanr(prompt_lengths, latencies)

    print(f'Spearman correlation coefficient: {correlation}')
    print(f'P-value: {p_value}')

# plot_length_vs_latency("gpt-4o")
# plot_length_vs_latency("gpt-3.5-turbo-0125")
