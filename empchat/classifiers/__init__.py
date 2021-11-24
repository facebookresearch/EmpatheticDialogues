def get_classifier_model(emo_model_type: str, model_path: str):
    if emo_model_type == "lstm":
        pass
    elif emo_model_type == "attn":
        pass
    elif emo_model_type == "trans":
        pass
    else:
        raise ValueError("'emo_model_type' supports only `lstm|attn|trans` values")
