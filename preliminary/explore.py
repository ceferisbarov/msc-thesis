long_prompts = []
with open("long_prompts.txt", "r") as file_:
    for line in file_.readlines():
        long_prompts.append(line.strip())

short_prompts = []
with open("short_prompts.txt", "r") as file_:
    for line in file_.readlines():
        short_prompts.append(line.strip())

avg_len_short = sum(map(len, short_prompts)) / len(short_prompts)
avg_len_long = sum(map(len, long_prompts)) / len(long_prompts)
print("Average length (long prompts):", avg_len_long)
print("Average length (short prompts):", avg_len_short)

