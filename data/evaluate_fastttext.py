import fasttext as fasttext_module

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

ftmodel = fasttext_module.FastText.load_model("../models/fasttext_empathetic_dialogues.mdl")

file_names = ["train", "valid", "test"]

for file_name in file_names:
    in_file = open(file_name + ".csv")
    correct = 0
    total = 0

    for idx, line in enumerate(in_file.readlines()):
        if idx == 0:
            continue
        else:
            splits = line.split(",")
            labels, _ = ftmodel.predict(splits[5].replace("_comma_", ","))
            if labels[0].split("_")[-1] not in label_map:
                print(labels)
            if labels[0].split("_")[-1] == splits[2]:
                correct += 1
            total += 1

    print("%s accuracy: %f" % (file_name, correct / total))
