from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore

import pyLDAvis
import pyLDAvis.gensim
import warnings
import _pickle as pickle
import nlp_utils as utils
import os
import spacy
import codecs
import itertools as it
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence


nlp = spacy.load('en')
intermediate_dir = os.path.join('..', 'intermediate')
bigram_model_filepath = os.path.join(intermediate_dir, 'bigram_model_all')
bigram_model = Phrases.load(bigram_model_filepath)
trigram_model_filepath = os.path.join(intermediate_dir, 'trigram_model_all')
trigram_model = Phrases.load(trigram_model_filepath)

def main():
    
    reviews_filepath = os.path.join(intermediate_dir, 'trigram_reviews_all.txt')
    '''
    Create dictionary
    '''
    trigram_dictionary_filepath = os.path.join(intermediate_dir, 'trigram_dict_all.dic')
    #createNgramDict(reviews_filepath, trigram_dictionary_filepath) ## comment when done
    trigram_dictionary = Dictionary.load(trigram_dictionary_filepath)
    
    
    '''
    Create bag of words
    '''
    trigram_bow_filepath = os.path.join(intermediate_dir, 'trigram_bow_corpus_all.mm')
    ## comment following line when generated
    #MmCorpus.serialize(trigram_bow_filepath,ngram_bow_generator(reviews_filepath, trigram_dictionary))
    trigram_bow_corpus = MmCorpus(trigram_bow_filepath)

    '''
    Create LDA model
    '''
    lda_model_filepath = os.path.join(intermediate_dir, 'lda_model_all')
    # comment when done
    #create_lda_model(trigram_bow_corpus, trigram_dictionary, lda_model_filepath, workers=3)
    lda = LdaMulticore.load(lda_model_filepath)

    # print topics
    exploreTopic(lda, topicn = 11)
    
    # graphically display topics
    LDAvis_prepared_filepath = os.path.join(intermediate_dir, 'lda_prepared')
    
    # comment when finished
    #ldavis_prepared = pyLDAvis.gensim.prepare(lda, trigram_bow_corpus, trigram_dictionary)
    # file open as Writing Binary mode
    #with open(LDAvis_prepared_filepath, 'wb') as f:
    #    pickle.dump(ldavis_prepared, f)
    
    
    #with open(LDAvis_prepared_filepath, 'rb') as f:
    #    ldavis_prepared = pickle.load(f)
    #    pyLDAvis.display(ldavis_prepared)
    
    review_number = 125
    review = get_sample_review(reviews_filepath, review_number)
    print('LDA topics for review number {}: \n {}'.format(review_number, review))
    lda_description(review, lda, trigram_dictionary)
    
def get_sample_review(reviews_filepath, review_number):
    """
    retrieve a particular review index
    from the reviews file and return it
    """
    return list(it.islice(utils.line_review(reviews_filepath),
                          review_number, review_number+1))[0]    
    
def lda_description(review_text, lda, dictionary, min_topic_freq=0.05):
    """
    accept the original text of a review and (1) parse it with spaCy,
    (2) apply text pre-proccessing steps, (3) create a bag-of-words
    representation, (4) create an LDA representation, and
    (5) print a sorted list of the top topics in the LDA representation
    """
    
    # parse the review text with spaCy
    parsed_review = nlp(review_text)
    
    # lemmatize the text and remove punctuation and whitespace
    unigram_review = [utils.preprocessLemma(token) for token in parsed_review
                      if not utils.punct_space(token)]
    
    # apply the first-order and secord-order phrase models
    bigram_review = bigram_model[unigram_review]
    trigram_review = trigram_model[bigram_review]
    
    # remove any remaining stopwords
    trigram_review = [term for term in trigram_review
                      if not term in spacy.en.STOP_WORDS]
    
    # create a bag-of-words representation
    review_bow = dictionary.doc2bow(trigram_review)
    
    # create an LDA representation
    review_lda = lda[review_bow]
    
    # sort with the most highly related topics first
    review_lda = sorted(review_lda, key=orderFunc)
    
    for topic_number, freq in review_lda:
        if freq < min_topic_freq:
            break
            
        # print the most highly related topic names and frequencies
        print('Topic number {}: {}'.format(topic_number, round(freq, 3)))

def orderFunc(lda_freq):
    #print(lda_freq)
    return -lda_freq[1]

def exploreTopic(lda, topicn, nterms = 25):
    
    print('{:20} {}'.format('term', 'frequency') + '\n')
    for term, frequency in lda.show_topic(topicn, topn = nterms):
        print('{:20} {:.3f}'.format(term, round(frequency,3)) )

def create_lda_model(corpus, dict, lda_model_filepath, workers=1, topics=50):
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