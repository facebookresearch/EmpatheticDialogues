from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
import torch
#archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz", cuda_device=0)
archive = load_archive("elmo-constituency-parser-2018.03.14.tar.gz", cuda_device=0)
predictor = Predictor.from_archive(archive, 'constituency-parser')


# Using readlines() 
file1 = open('uniquesentences_unstripped.txt', 'r') 
cands = file1.readlines() 
to_predict = []
tags = []
batch_len = 250
with torch.no_grad():
    for index, cand in enumerate(cands):
        curr_dict = {"sentence": cand.replace("_comma_",", ")}
        to_predict.append(curr_dict)
                        #z = predictor.predict_json()

        if len(to_predict)==batch_len:
            print(index)
            batch_answer = predictor.predict_batch_json(to_predict)
            to_predict = []
            for answer in batch_answer:
                tags.append(answer['trees'])
            
            z = 1

with open('constituency_trees.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % sentence for sentence in tags)
