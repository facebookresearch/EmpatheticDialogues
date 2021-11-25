from typing import List

class Instance:
    words: List[str]
    ori_sentence: str
    label: str = None
    prediction: str = None

    def __init__(self, sentence: str, words: List[str], label: str = None):
        self.ori_sentence = sentence
        self.words = words
        self.label = label
