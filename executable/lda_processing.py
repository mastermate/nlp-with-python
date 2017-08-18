from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore

import pyLDAvis
import pyLDAvis.gensim
import warnings
import cPickle as pickle
import os
import spacy
import codecs
import itertools as it
from gensim.models.word2vec import LineSentence



def main():
    intermediate_dir = os.path.join('..', 'intermediate')
    reviews_filepath = os.path.join(intermediate_dir, 'trigram_reviews_all.txt')
    
    '''
    Create dictionary
    '''
    trigram_dictionary_filepath = os.path.join(intermediate_dir, 'trigram_dict_all.dic')
    createNgramDict(reviews_filepath, trigram_dictionary_filepath) ## comment when done
    trigram_dictionary = Dictionary.load(trigram_dictionary_filepath)
    
    
    '''
    Create bag of words
    '''
    trigram_bow_filepath = os.path.join(intermediate_dir, 'trigram_bow_corpus_all.mm')
    ## comment following line when generated
    MmCorpus.serialize(trigram_bow_filepath,ngram_bow_generator(reviews_filepath, trigram_dictionary))
    trigram_bow_corpus = MmCorpus(trigram_bow_filepath)

    '''
    Create LDA model
    '''
    lda_model_filepath = os.path.join(intermediate_dir, 'lda_model_all')
    # comment when done
    create_lda_model(trigram_bow_corpus, trigram_dictionary, 3, 50)
    lda = LdaMulticore.load(lda_model_filepath)

def create_lda_model(corpus, dict, workers, topics=50):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        ## workers sets the parallelism,
        ## and should be the number of physical cores - 1
        lda = LdaMulticore(corpus, num_topics = topics, id2word=dict, workers=workers)
        lda.save(lda_model_filepath)
    

def ngram_bow_generator(filepath, dictionary):
    '''
    Generator function to read reviews from a file
    and yield a bag of words representation
    '''
    for review in LineSentence(filepath):
        yield dictionary.doc2bow(review)

def createNgramDict(reviews_filepath, dictionary_filepath):
    reviews = LineSentence(reviews_filepath)
    # learn dictionary
    dict = Dictionary(reviews)

    #filter and compactify
    dict.filter_extremes(no_below=10, no_above=0.4)
    dict.compactify()
    
    dict.save(dictionary_filepath)
    
    
    
if __name__ == '__main__':
    main()