import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download("wordnet")
import re


def remove_punctuation(sentence: str): return re.sub(r'[^a-zA-Z0-9 ]', '', sentence)


def remove_stopwords(words_list):
    # words_list = word_tokenize(remove_punctuation(sentence))
    return list(filter(lambda x: x not in stopwords.words('english'), words_list))


def lemmatize(word_list):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return list(map(lemmatizer.lemmatize, word_list))


def tokenize(text):
    return lemmatize(remove_stopwords(word_tokenize(remove_punctuation(text.lower()))))