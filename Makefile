REDDIT_EMBEDDINGS_PATH = ./reddit
EMPATHETIC_DIALOGUES_DATA_FOLDER = ./empatheticdialogues
PRETRAINED_MODEL_PATH = ./pretrained
TRAIN_SAVE_FOLDER = ./train
EVAL_SAVE_FOLDER = ./evaluation
REDDIT_DATA_FOLDER = ./reddit/data

Pretraining:
	python retrieval_train.py \
        --batch-size 512 \
        --cuda \
        --dataset-name reddit \
        --dict-max-words 250000 \
        --display-iter 250 \
        --embeddings ${REDDIT_EMBEDDINGS_PATH} \
        --empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
        --learn-embeddings \
        --learning-rate 8e-4 \
        --model transformer \
        --model-dir ${TRAIN_SAVE_FOLDER} \
        --model-name model \
        --n-layers 4 \
        --num-epochs 10000 \
        --optimizer adamax \
        --reddit-folder ${REDDIT_DATA_FOLDER} \
        --transformer-dim 300 \
        --transformer-n-heads 6

Fine-tuning:
	python retrieval_train.py \
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
        --model-name model \
        --n-layers 4 \
        --num-epochs 10 \
        --optimizer adamax \
        --reddit-folder ${REDDIT_DATA_FOLDER} \
        --transformer-dim 300 \
        --transformer-n-heads 6

Evaluation:
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
        --model-name model \
        --n-layers 4 \
        --optimizer adamax \
        --pretrained ${TRAIN_SAVE_FOLDER}/model.mdl \
        --reactonly \
        --transformer-dim 300 \
        --transformer-n-heads 6

# BLEU (EmpatheticDialogues context/candidates)
BLEU_Evaluation:
	python retrieval_eval_bleu.py \
        --empchat-cands \
        --empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
        --max-hist-len 4 \
        --model ${TRAIN_SAVE_FOLDER}/model.mdl \
        --name model \
        --output-folder ${EVAL_SAVE_FOLDER} \
        --reactonly \
        --task empchat

BERT_Pretraining:
	python retrieval_train.py \
        --batch-size 256 \
        --bert-dim 300 \
        --cuda \
        --dataset-name reddit \
        --dict-max-words 250000 \
        --display-iter 100 \
        --embeddings None \
        --empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
        --learning-rate 6e-5 \
        --model bert \
        --model-dir ${TRAIN_SAVE_FOLDER} \
        --model-name model \
        --num-epochs 10000 \
        --optimizer adamax \
        --reddit-folder ${BERT_TOKENIZED_REDDIT_DATA_FOLDER}

BERT_Fine-tuning:
	python retrieval_train.py \
        --batch-size 256 \
        --bert-dim 300 \
        --cuda \
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
        --model-name model \
        --num-epochs 100 \
        --optimizer adamax \
        --stop-crit-num-epochs 10


# P@1,100
BERT_Evaluation:
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
        --model-name model \
        --optimizer adamax \
        --pretrained ${TRAIN_SAVE_FOLDER}/model.mdl \
        --reactonly

# BLEU (EmpatheticDialogues context/candidates)
BERT_BLEU:
	python retrieval_eval_bleu.py \
        --bleu-dict ${PATH_TO_MODEL_WITH_TRANSFORMER_DICT} \
        --empchat-cands \
        --empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
        --max-hist-len 4 \
        --model ${TRAIN_SAVE_FOLDER}/model.mdl \
        --name model \
        --output-folder ${EVAL_SAVE_FOLDER} \
        --reactonly \
        --task empchat




