from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

def has_help(x): return 'help' in x
def has_need(x): return 'need' in x

class TextExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, word_to_find):
        self.word = word_to_find
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(list(map(lambda x: self.word in x, X)))

#
# class HelpExtractor(BaseEstimator, TransformerMixin):
#     """ Returns true when 'help' in in the text """
#     def __init__(self):
#         pass
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         return pd.DataFrame(list(map(has_help, X)))
#
# class NeedExtractor(BaseEstimator, TransformerMixin):
#     """ Returns true when 'need' in in the text """
#     def __init__(self):
#         pass
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         return pd.DataFrame(list(map(has_need, X)))