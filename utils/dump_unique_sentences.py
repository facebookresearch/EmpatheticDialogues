import csv
import os
# https://github.com/facebookresearch/EmpatheticDialogues/issues/21
# https://github.com/facebookresearch/EmpatheticDialogues/issues/29
# https://github.com/facebookresearch/EmpatheticDialogues/blob/master/empchat/datasets/empchat.py

"""
with open('empatheticdialogues/test.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        print(row)
"""

import nltk


data_folder = "empatheticdialogues"
I = 8 # Candidate utterence excel column label
F = 5 # Actual utterence from initaitor and the listener
splits = ["train", "test", "valid"]
sentence_set = set()
dummy_list = []

for splitname in splits:
    print(splitname)
    df = open(os.path.join(data_folder, f"{splitname}.csv")).readlines()

    for i in range(1, len(df)):
        if i%100==0:
            print(i)
        row = df[i].split(",")
        candidate_utterences = []
        candidate_utterence_sentences = []
        utterence = row[F]

        if len(row)==9:
            candidate_utterences = row[I].split("|")

        utterence_sentences = nltk.sent_tokenize(utterence) 
        for candidate_utterence in candidate_utterences:
            candidate_utterence_tokenized = nltk.sent_tokenize(utterence)
            for sentence in candidate_utterence_tokenized:
                candidate_utterence_sentences.append(sentence)

        fin = utterence_sentences+candidate_utterence_sentences 
        for sentence in fin:
            sentence_set.add(sentence)
sentence_list = list(sentence_set)
print(len(sentence_list))
        
with open('uniquesentences_unstripped.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % sentence for sentence in sentence_list)

        #sent_text = nltk.sent_tokenize(text) #
