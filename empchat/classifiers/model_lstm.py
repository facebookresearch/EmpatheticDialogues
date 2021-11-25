from empchat.datasets.tokens import get_bert_token_mapping

if __name__ == "__main__":
    from data_loader import EmotionDataset
    from torch.utils.data import DataLoader
    from utils import build_word_idx, PAD
    from empchat.datasets.loader import pad
    from pytorch_pretrained_bert import BertTokenizer

    # TODO: provide tokenization function
    tokenizer = BertTokenizer.from_pretrained(
                    "bert-base-cased",
                    do_lower_case=False,
                    never_split=(
                        ["[CLS]", "[MASK]"]
                        + list(get_bert_token_mapping(None).values())
                    ),
                )

    # load data
    train_dataset = EmotionDataset("../../data/train.csv", True, tokenizer.tokenize)
    valid_dataset = EmotionDataset("../../data/valid.csv", False, tokenizer.tokenize, label2idx=train_dataset.label2idx)
    test_dataset = EmotionDataset("../../data/test.csv", False, tokenizer.tokenize, label2idx=train_dataset.label2idx)

    word2idx, idx2word, char2idx, idx2char = build_word_idx(
        train_dataset.insts, valid_dataset.insts, test_dataset.insts
    )

    # convert to tensors
    train_dataset.convert_instances_to_feature_tensors(word2idx)
    valid_dataset.convert_instances_to_feature_tensors(word2idx)
    test_dataset.convert_instances_to_feature_tensors(word2idx)


    def batchify(batch):
        input_list = list(zip(*batch))
        contexts, next_ = [
            pad(ex, word2idx[PAD]) for ex in [input_list[0], input_list[1]]
        ]
        return contexts, next_


    # create loaders
    train_loader = DataLoader(
        train_dataset,
        # TODO: set from CMD
        batch_size=16,
        shuffle=True,
        num_workers=12,
        # see if this works
        collate_fn=batchify,
        pin_memory=False,
    )
