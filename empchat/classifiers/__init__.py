import json


class DummyModel:
    def __init__(self, model_name, splitname):
        self.data = json.load(open("data/%s/%s.json" % (model_name, splitname)))

    def predict(self, sentence: str, k: int = 1):
        return [self.data[sentence]], None


def get_classifier_model(emo_model_type: str, splitname: str):
    if emo_model_type in ["lstm", "attn", "trans"]:
        return DummyModel(emo_model_type, splitname)
    else:
        raise ValueError("'emo_model_type' supports only `lstm|attn|trans` values")
