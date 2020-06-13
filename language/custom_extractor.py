from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

def has_help(x): return 'help' in x
def has_need(x): return 'need' in x

class TextExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, word_to_find):
        self.word = word_to_find.lower()
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(list(map(lambda x: self.word in x.lower(), X)))


class LenExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, word_to_find):
        self.word = word_to_find.lower()
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(list(map(lambda x: len(x), X)))

