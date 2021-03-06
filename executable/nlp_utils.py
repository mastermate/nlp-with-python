import spacy
import codecs

'''
Aux functions
'''

def preprocessLemma(token):
    '''
    Preprocess a lemma to skip the new -PRON- category in spacy
    '''
    if token.lemma_ == '-PRON-':
        return token.lower_
    return token.lemma_

def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """
    return token.is_punct or token.is_space

def line_review(filename):
    """
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """
    with codecs.open(filename, encoding='utf_8') as f:
        for review in f:
            yield review.replace('\\n', '\n')
            
def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """
    for parsed_review in nlp.pipe(line_review(filename), batch_size=10000, n_threads=4):
        for sent in parsed_review.sents:
            yield u' '.join([preprocessLemma(token) for token in sent if not punct_space(token)])
            



if __name__ == '__main__':
    pass