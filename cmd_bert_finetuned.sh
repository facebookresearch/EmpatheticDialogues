EMPATHETIC_DIALOGUES_DATA_FOLDER=data
EVAL_SAVE_FOLDER=eval_save
TRAIN_SAVE_FOLDER=train_save

python retrieval_train.py \
--batch-size 256 \
--bert-dim 300 \
--cuda \
--dataset-name empchat \
--dict-max-words 250000 \
--display-iter 100 \
--embeddings None \
--empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
--max-hist-len 4 \
--model bert \
--model-dir ${EVAL_SAVE_FOLDER} \
--model-name bert_finetuned \
--optimizer adamax \
--pretrained models/bert_finetuned.mdl \
--reactonly