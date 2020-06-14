"""
This module defines some transformer to extract features from a text message
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TextExtractor(BaseEstimator, TransformerMixin):
    """    Returns True if `word_to_find` is in the message    """
    def __init__(self, word_to_find):
        self.word = word_to_find.lower()
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(list(map(lambda x: self.word in x.lower(), X)))


class LenExtractor(BaseEstimator, TransformerMixin):
    """    Returns the length of the message    """
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(list(map(lambda x: len(x), X)))

