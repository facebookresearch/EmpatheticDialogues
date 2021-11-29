import json


class DummyModel:
    def __init__(self, model_name, splitname, label_suffix):
        self.data = json.load(open("data/%s/%s%s.json" % (model_name, splitname, label_suffix)))
        self.data4 = json.load(open("data/%s/%s%s-4.json" % (model_name, splitname, label_suffix)))

    def predict(self, sentence: str, k: int = 1):
        if sentence in self.data:
            return [self.data[sentence]], None
        else:
            return [self.data4[sentence]], None


def get_classifier_model(emo_model_type: str, splitname: str, label_suffix: str):
    if emo_model_type in ["lstm", "attn", "trans"]:
        return DummyModel(emo_model_type, splitname, label_suffix)
    else:
        raise ValueError("'emo_model_type' supports only `lstm|attn|trans` values")
