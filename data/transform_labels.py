label_map = {
    # 1
    "angry": "anger",
    "annoyed": "anger",
    "furious": "anger",
    "jealous": "anger",
    # 2
    "surprised": "surprise",
    "impressed": "surprise",
    # 3
    "anticipating": "anticipation",
    "hopeful": "anticipation",
    # 4
    "excited": "joy",
    "proud": "joy",
    "joyful": "joy",
    "content": "joy",
    "confident": "joy",
    "prepared": "joy",
    # 5
    "sad": "sadness",
    "lonely": "sadness",
    "guilty": "sadness",
    "disappointed": "sadness",
    "embarrassed": "sadness",
    "ashamed": "sadness",
    "nostalgic": "sadness",
    "sentimental": "sadness",
    "devastated": "sadness",
    # 6
    "trusting": "trust",
    "caring": "trust",
    "faithful": "trust",
    "grateful": "trust",
    # 7
    "disgusted": "disgusted",
    # 8
    "afraid": "fear",
    "terrified": "fear",
    "anxious": "fear",
    "apprehensive": "fear",
}

file_names = ["train", "valid", "test"]
final_set = set(label_map.values())

for file_name in file_names:
    in_file = open(file_name + ".csv")
    out_file = open(file_name + "_%d.csv" % len(final_set), "w")

    for idx, line in enumerate(in_file.readlines()):
        if idx == 0:
            out_file.write(line)
        else:
            splits = line.split(",")
            splits[2] = label_map[splits[2]]
            out_file.write(",".join(splits))

    in_file.close()
    out_file.close()
