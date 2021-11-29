from collections import Counter

file_names = ["train", "valid", "test"]
counter = Counter()

for file_name in file_names:
    out_file = open(file_name + ".csv")
    # out_file = open(file_name + "_8.csv")
    for idx, line in enumerate(out_file.readlines()):
        if idx == 0:
            continue
        else:
            splits = line.split(",")
            counter[splits[2]] += 1

print(counter)
total = sum(counter.values())
print(total)
print(dict(map(lambda x: (x[0], x[1] / total), counter.items())))
