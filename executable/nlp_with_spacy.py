import spacy
import os
import codecs
import pandas as pd
import itertools as it

def main():
    playWithSampleReview()
    
def playWithSampleReview():
    # loading spacy english dic
    nlp = spacy.load('en')

    # opening the sample review file
    review_txt_filepath = os.path.join('..','intermediate', 'sample_review.txt')
    with codecs.open(review_txt_filepath, encoding='utf_8') as f:
        sample_review = f.read()
    print(sample_review)

    # playing with Spacy
    
    # 1. sentences decomposition
    parsed_review = nlp(sample_review)
    for num, sentence in enumerate(parsed_review.sents):
        print('Sentence ',num + 1, ' :')
        print(sentence)
    
    # 2. entity recognition
    for num, entity in enumerate(parsed_review.ents):
        print('Entity ', num + 1, ':', entity, '-', entity.label_)
  
if __name__ == '__main__':
    main()