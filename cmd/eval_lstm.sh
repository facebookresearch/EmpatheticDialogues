EMPATHETIC_DIALOGUES_DATA_FOLDER=data
MODEL_NAME=bert_ft_lstm
EVAL_SAVE_FOLDER=models/${MODEL_NAME}/eval_save
TRAIN_SAVE_FOLDER=models/${MODEL_NAME}/train_save
PATH_TO_MODEL_WITH_TRANSFORMER_DICT=${TRAIN_SAVE_FOLDER}/${MODEL_NAME}.mdl
PATH_TO_TRAINED_FASTTEXT_MODEL=dummy

EMO_MODEL=lstm python retrieval_eval_bleu.py \
--bleu-dict ${PATH_TO_MODEL_WITH_TRANSFORMER_DICT} \
--empchat-cands \
--empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
--max-hist-len 4 \
--model ${TRAIN_SAVE_FOLDER}/${MODEL_NAME}.mdl \
--name ${MODEL_NAME} \
--output-folder ${EVAL_SAVE_FOLDER}_bleu \
--reactonly \
--task empchat \
--fasttext 1 \
--fasttext-path ${PATH_TO_TRAINED_FASTTEXT_MODEL} \
--fasttext-type emo


EMO_MODEL=lstm python retrieval_train.py \
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
--model-name ${MODEL_NAME} \
--optimizer adamax \
--pretrained ${TRAIN_SAVE_FOLDER}/${MODEL_NAME}.mdl \
--reactonly \
--fasttext 1 \
--fasttext-path ${PATH_TO_TRAINED_FASTTEXT_MODEL} \
--fasttext-type emo