import os
import codecs
import spacy
import nlp_utils as utils
import itertools as it
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

nlp = spacy.load('en')

def main():
    intermediate_directory = os.path.join('..','intermediate')
    unigram_sentences_filepath = os.path.join(intermediate_directory, 'unigram_sentences_all.txt') 
    sample_review_filepath = os.path.join(intermediate_directory, 'some_reviews.txt') 
    #createUnigramFile(sample_review_filepath, unigram_sentences_filepath)
    
    '''
    Unigrams
    '''
    print(' ')
    print('----------- Unigrams -----------')
    print(' ')
    unigram_sentences = LineSentence(unigram_sentences_filepath)
    for unigram_sentence in it.islice(unigram_sentences, 4, 8):
        print(' '.join(unigram_sentence))
        print ('')
        
    '''
    Bigrams
    '''
    bigram_model_filepath = os.path.join(intermediate_directory, 'bigram_model_all')
    #createNgramModel(unigram_sentences, bigram_model_filepath) ## comment this line when bigrams file is created
    bigram_model = Phrases.load(bigram_model_filepath)
    
    bigram_sentences_filepath = os.path.join(intermediate_directory, 'bigram_sentences_all.txt')
    createNgramFile(unigram_sentences, bigram_model, bigram_sentences_filepath)
    
    print(' ')
    print('----------- Bigrams -----------')
    print(' ')
    bigram_sentences = LineSentence(bigram_sentences_filepath)
    for bigram_sentence in it.islice(bigram_sentences, 4, 8):
        print(' '.join(bigram_sentence))
        print('')
        
    '''
    Trigrams
    '''
    trigram_model_filepath = os.path.join(intermediate_directory, 'trigram_model_all')
    createNgramModel(bigram_sentences, trigram_model_filepath)
    trigram_model = Phrases.load(trigram_model_filepath)
    
    trigram_senteces_filepath = os.path.join(intermediate_directory, 'trigram_sentences_all.txt')
    createNgramFile(bigram_sentences, trigram_model, trigram_senteces_filepath)
    
    print(' ')
    print('----------- Trigrams -----------')
    print(' ')
    trigram_sentences = LineSentence(trigram_senteces_filepath)
    for trigram_sentence in it.islice(trigram_sentences, 4, 8):
        print(' '.join(trigram_sentence))
        print('')
        
    '''
    Reviewed trigrams.
    Note: we have to run the last part again, since pronouns appear as -PRON-
    I forgot to call the nlp_utils func to avoid that :s
    '''
    reviewed_trigrams_filepath = os.path.join(intermediate_directory, 'trigram_reviews_all.txt')
    with codecs.open(reviewed_trigrams_filepath, 'w', encoding='utf_8') as f:
        for parsed_review in nlp.pipe(line_review(sample_review_filepath), batch_size=10000, n_threads=4):
            # lemmatise review, removing punctuation
            unigram_review = [utils.preprocessLemma(token) for token in parsed_review if not punct_space(token)]
            
            # first and second order phrase models
            bigram_review = bigram_model[unigram_review]
            trigram_review = trigram_model[bigram_review]
            
            # remove remaining stopwords
            trigram_review = [term for term in trigram_review if term not in spacy.en.STOP_WORDS]
            
            # write the result review as a line in the new file
            f.write(' '.join(trigram_review) +  '\n')

def createNgramModel(ngram_sents, outputFilePath):
    bigram_model = Phrases(ngram_sents)
    bigram_model.save(outputFilePath)
    
def createNgramFile(n_1gram_sents, ngram_model, outputFilePath):
    with codecs.open(outputFilePath, 'w', encoding='utf_8') as f:
        for n_1gram_sent in n_1gram_sents:
            ngram_sentence = u' '.join(ngram_model[n_1gram_sent])
            f.write(ngram_sentence + '\n')

def createUnigramFile(inputFilePath, outputFilePath):
    with codecs.open(outputFilePath, 'w', encoding = 'utf_8') as f:
        for sent in lemmatized_sentence_corpus(inputFilePath):
            f.write(sent + '\n')


'''
Aux functions
'''

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
            yield u' '.join([utils.preprocessLemma(token) for token in sent if not punct_space(token)])
            

if __name__ == '__main__':
    main()