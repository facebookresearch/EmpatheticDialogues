EMPATHETIC_DIALOGUES_DATA_FOLDER=data
EVAL_SAVE_FOLDER=eval_save
TRAIN_SAVE_FOLDER=train_save
PRETRAINED_MODEL_PATH=models/normal_transformer_pretrained.mdl
REDDIT_DATA_FOLDER=reddit
PATH_TO_TRAINED_FASTTEXT_MODEL=dummy

EMO_MODEL=lstm LABEL_SUFFIX=_8 python retrieval_train.py \
--batch-size 512 \
--cuda \
--dataset-name empchat \
--dict-max-words 250000 \
--display-iter 250 \
--empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
--learn-embeddings \
--learning-rate 8e-4 \
--load-checkpoint ${PRETRAINED_MODEL_PATH} \
--max-hist-len 4 \
--model transformer \
--model-dir ${TRAIN_SAVE_FOLDER} \
--model-name normal_transformer_pretrained \
--n-layers 4 \
--num-epochs 10 \
--optimizer adamax \
--reddit-folder ${REDDIT_DATA_FOLDER} \
--transformer-dim 300 \
--transformer-n-heads 6 \
--fasttext 1 \
--fasttext-path ${PATH_TO_TRAINED_FASTTEXT_MODEL} \
--fasttext-type emo
