EMPATHETIC_DIALOGUES_DATA_FOLDER=data
MODEL_NAME=bert_ft_lstm_attn
EVAL_SAVE_FOLDER=models/${MODEL_NAME}/eval_save
TRAIN_SAVE_FOLDER=models/${MODEL_NAME}/train_save
PRETRAINED_MODEL_PATH=models/bert_pretrained.mdl
PATH_TO_TRAINED_FASTTEXT_MODEL=dummy

EMO_MODEL=attn python retrieval_train.py \
--batch-size 256 \
--cuda \
--bert-dim 300 \
--dataset-name empchat \
--dict-max-words 250000 \
--display-iter 100 \
--embeddings None \
--empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
--learning-rate 1e-5 \
--load-checkpoint ${PRETRAINED_MODEL_PATH} \
--max-hist-len 4 \
--model bert \
--model-dir ${TRAIN_SAVE_FOLDER} \
--model-name ${MODEL_NAME} \
--num-epochs 15 \
--optimizer adamax \
--stop-crit-num-epochs 10 \
--fasttext 1 \
--fasttext-path ${PATH_TO_TRAINED_FASTTEXT_MODEL} \
--fasttext-type emo