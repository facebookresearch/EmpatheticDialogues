EMPATHETIC_DIALOGUES_DATA_FOLDER=data
EVAL_SAVE_FOLDER=eval_save
TRAIN_SAVE_FOLDER=train_save

python retrieval_eval_bleu.py \
--empchat-cands \
--empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
--max-hist-len 4 \
--model models/normal_transformer_pretrained.mdl \
--name normal_transformer_pretrained \
--output-folder ${EVAL_SAVE_FOLDER} \
--reactonly \
--task empchat