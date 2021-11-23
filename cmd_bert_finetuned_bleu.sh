EMPATHETIC_DIALOGUES_DATA_FOLDER=data
EVAL_SAVE_FOLDER=eval_save
PATH_TO_MODEL_WITH_TRANSFORMER_DICT=models/normal_transformer_finetuned.mdl

python retrieval_eval_bleu.py \
--bleu-dict ${PATH_TO_MODEL_WITH_TRANSFORMER_DICT} \
--empchat-cands \
--empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
--max-hist-len 4 \
--model models/bert_finetuned.mdl \
--name bert_finetuned \
--output-folder ${EVAL_SAVE_FOLDER} \
--reactonly \
--task empchat