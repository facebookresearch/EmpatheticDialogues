EMPATHETIC_DIALOGUES_DATA_FOLDER=data
EVAL_SAVE_FOLDER=eval_save
TRAIN_SAVE_FOLDER=train_save

python retrieval_train.py \
--batch-size 512 \
--cuda \
--dataset-name empchat \
--dict-max-words 250000 \
--display-iter 250 \
--empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
--max-hist-len 4 \
--model transformer \
--model-dir ${EVAL_SAVE_FOLDER} \
--model-name normal_transformer_pretrained \
--n-layers 4 \
--optimizer adamax \
--pretrained models/normal_transformer_pretrained.mdl \
--reactonly \
--transformer-dim 300 \
--transformer-n-heads 6
