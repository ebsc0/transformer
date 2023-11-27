import spacy

from transformer import *

spacy_en = spacy.load('en_core_web_sm')
spacy_fr = spacy.load('fr_core_news_sm')

def dataset():
    raise NotImplementedError

def train():
    raise NotImplementedError

def evaluate():
    raise NotImplementedError

def main():
    model = Transformer()

if __name__ == "__main__":
    main()