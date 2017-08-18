import spacy

def preprocessLemma(token):
    if token.lemma_ == '-PRON-':
        return token.lower_
    return token.lemma_

if __name__ == '__main__':
    pass