###########################################################################################################################
## Akshay Surendra Phadnis
## 22 October 2015
## NET-ID: asp150630

## enhanced_query_generation_with_synsets.py is
## uses viterbi algorithm to tag each word in a question and it's synonyms as either a key word  (denoted by the tag K )or a non keyword
## (denoted by the tag NK) the does part of speech tagging to filter the partial query and after that, does further
#filtering to retain words pertaining to the closed domain
#Please refer ProjectReport-AKSHAY_PHADNIS.pdf for details about this project.

#This exaustive enumeration over synonyms is slow and does not yield results quickly

###########################################################################################################################
from nltk.stem.lancaster import LancasterStemmer
from nltk import pos_tag , word_tokenize
from nltk.corpus import wordnet as wn
from pprint import pprint
#pprint renders readable & line by line prints of lists and dictionaries
##Due to corpora containing some unicode data, warnings may be produced
##so as to supress them, the following two lines have been used. They may be commented
import warnings
warnings.filterwarnings("ignore")

st = LancasterStemmer()
def viterbi(x,q,e, initial_state_probabilities):
    """
    This function takes the following input:
    
    x is the sequence (list) of observations x1 through xn
    q is a dictionary of the format  (v,u) : P(v|u), where u and v are hidden states
    e is a dictionary of the format  (x,v) : P(x|v), where x is an observation and v is the
    corresponding hidden state
    initial_state_probabilities is a dictionary bearing states and their intital probabilities

    then, executes the Viterbi algorithm

    and returns a list y1...yn of hidden states

    Note: All hidden states and observations are strings
    
    """
    
    
    S = set( [] )
    #Garnering set of hidden states
    for state1,state2 in q.keys():
        S.add(state1)
        S.add(state2)
        
    S.add('*')#adding a dummy start  state
    
    
    pt = {}
    #pt would be a dictionary of the form:
    #(k,v): maximum probility of hidden state sequence ending in v at position k

    
    pt[(0,'*')] = 1.0
    #for all hidden states other than *, pt[(0,v)] = 0
    for v in S :
        if v != '*':
            pt[(0,v)] = 0.0

            
    #for every observation, emission probability from dummy start state = 0.0        
    for observation in x:
        e[observation,'*']=0.0

    #Setting  transition probabilities involving initial state *
    for state in S:
        #Given state, probability of being in the same state is 1.0
         q[('*','*')] = 1.0##for all other state,state pairs, probabilities are already in q
         if state != '*':
             q[(state,'*')]  = initial_state_probabilities[state]
             q[('*',state)] = 0.0
             #transition from hot or cold back to * cannot happen and hence, transition probability is 0 
         
         
    b = {}
    #b is the table  used for pointing back
    #and is a dictionary of the form:
    #(k,v): the state u immediately before v for which  probility of hidden state sequence ending in v at
    #position k is maximum

    running_sum_of_joint_probabilities = 0 
    n  = len(x)
    for k in range(1,n+1):# for k in 1 to n
        for v in S:
            pt[(k, v)], b[(k,v)]  = max( [(pt[(k-1 , u)] * q[(v,u)]  * e[(x[k-1],v)],u) for u in S ] )
            
                        
                       
        
    #Obtaining hidden states from the last state to the first state:

    #procuring last hidden state
    y = []
    yn = max([(pt[(n, v)],v) for v in S  ])[1]#last hidden state is the one for which pt[(n,v)] is highest
    y.append(yn)

    #procuring other  hidden states from yn-1 to y1
    for k in range(n-1,1-1,-1):
       y.append(b[ ( k+1, y[n-(k+1)] ) ]) # k+1 is being subtracted from n since
       # (k+1)th ys are being obtained in reverse order

    #pprint(q)
    #pprint(pt)
       
    #returning and re-arranging states from first state to last state
    #print x
    
    return [y[len(y)-1-i]for i in range(len(y))]
    






def preprocess(x):
        """
        This helper function
        takes a question i.e. a string as an input and returns the string
        after removing punctuations and changing letters to lower case
        """
        return x.strip('?').lower()

def is_the_query_relevent_LESK(list_of_words_related_corpus, list_of_words_unrelated_corpus, set_of_words_in_query ):
        """
        Takes as input:
        List of words in related corpus
        List of words in unrelated corpus
        Set of words in the query
        
        and outputs a boolean indicating if the query is relevant to the related corpus
        
        """
        N = 2 #number of senses = 2 since a word can either belong to the corpus or not i.e. one sense is that the word
        # belongs to the corpus, the other is that it does not
        set_of_stemmed_words_in_query = set( st.stem(w) for w in set_of_words_in_query )
        overlaps_senses = []
        for idx in range(N):
            ##Do the following for each sense:
            if idx == 1:
                sense = "K" #Keyword
                set_of_words_in_gloss_and_examples  = list_of_words_related_corpus
            elif idx == 0:
                sense  = "NK" #not a keyword
                set_of_words_in_gloss_and_examples =  list_of_words_unrelated_corpus 

                
            ##Here, idx stands for index
            ##and since idx 0 is assumed to have more probability
            ##lower idx or higher N - idx  means high probability
    
            ## compute intersection of set(sense_data) with set(sentence)
            common_words = set(set_of_words_in_gloss_and_examples).intersection( set(set_of_words_in_query ))
            common_words_with_stemming = set(set_of_words_in_gloss_and_examples).intersection( set(set_of_stemmed_words_in_query ))
            
            overlaps_senses.append( (max(len(common_words), len(common_words_with_stemming)) ,sum( [set_of_words_in_gloss_and_examples.count(w) for w in set_of_words_in_query] ), sense) )
            ##Senses are chosen based on length of intersection, and if ties happen,
            ##they are broken based on probability

        
        
        ##return true if the question is relevant i.e. if it has the maximum overlap with the
        ##related corpus
        return max(overlaps_senses)[2] == "K"


## Probabilites for the Hidden Markov Model:

#transition probabilities
q ={
    ("K","K"): 0.7,
    ("K","NK"): 0.3,
    ("NK","K"): 0.3,
    ("NK","NK"): 0.7
}


initial_state_probabilities = {
    "K": 0.5,
    "NK": 0.5
    }
#A given word may or may not be a key-word
# thus, the probaility of this has been assumed to be equal
# i.e. the intial distribution is considerred as uniform



##emission probabilities
def obtain_emission_probailities(list_of_words_in_question, filename_corpus1, filename_corpus2 ):
    """
    Input: Question for the QA system (a string), filename of related corpus as the second argument
    filename of second corpus as the third argument
        
    Purpose: generating emision probabilities for the HMM considering each word from the question as an evidence
    
    Output: a dictionary of the format  (x,v) : P(x|v), where x is an observation and v is the corresponding hidden state
    i.e. tag K or NK
    """
    
    from string import punctuation
    e = {}

    #reading lines from related corpus
    file_containing_related_corpus = open(filename_corpus1, 'r')
    related_corpus =  file_containing_related_corpus.readlines()
    file_containing_related_corpus.close()

    #creating a list of words in related corpus
    list_of_words_in_related_corpus = []
    for line in related_corpus:
       list_of_words = line.split(' ')
       list_of_words = [ word.strip(punctuation).lower().strip('\n')  for word in list_of_words]
       list_of_words_in_related_corpus.extend( list_of_words )
       
    #reading lines from unrelated corpus
    file_containing_unrelated_corpus = open(filename_corpus2, 'r')
    unrelated_corpus =  file_containing_unrelated_corpus.readlines()
    file_containing_unrelated_corpus.close()

    #creating a list of words in unrelated corpus
    list_of_words_in_unrelated_corpus = []
    for line in unrelated_corpus:
       list_of_words = line.split(' ')
       list_of_words = [ word.strip(punctuation).lower().strip('\n')   for word in list_of_words]
       list_of_words_in_unrelated_corpus.extend( list_of_words )
       
    
    #number of words in related corpus
    n_r = float(len(list_of_words_in_related_corpus))

    #number of words in nrelatde corpus
    n_ur = float(len(list_of_words_in_unrelated_corpus))
    for w in list_of_words_in_question:
        w_stem = st.stem(w)
        #emission probability for the state 'keyword' is the maximum between frequency of word ant it's stem
        e[(w ,"K")] =  max(list_of_words_in_related_corpus.count(w)/n_r, list_of_words_in_related_corpus.count(w_stem)/n_r )
        if e[(w, "K")] == 0.0 : e[(w, "K")] = 0.0000000000000000002 #since zero probabilities will eventually cause the joint score
        #to be zero despite the presence of keywords, the emission probability in such a case is kept low, but non-zero

        #emission probability for the state 'not keyword' is the maximum between frequency of word ant it's stem
        e[(w ,"NK")] =  max(list_of_words_in_unrelated_corpus.count(w)/n_ur, list_of_words_in_unrelated_corpus.count(w_stem)/n_ur )
        if e[(w, "NK")] == 0.0 : e[(w, "NK")] = 0.0000000002#since zero probabilities will eventually cause the joint score
        #to be zero despite the presence of keywords, the emission probability in such a case is kept low, but not zero
        #Also, due to this, if a word does not appear in any corpus, that word will be treated as a non keyword

         
    return (e, list_of_words_in_related_corpus, list_of_words_in_unrelated_corpus)



def obtain_synonyms(word):
	list_of_synsets = wn.synsets(word)
	synonyms = []
	for synset in list_of_synsets:
		for lemma in synset.lemmas():
			 synonyms.append(lemma.name())
	return synonyms

##1. Procuring a question
print "Please type in your question ?"    
question = raw_input()

##Preprocessing the question
question = preprocess(question)
list_of_words_in_question = question.split()

#Computation of emission probabilites for the Hidden Markov Model
e, list_of_words_related_corpus, list_of_words_unrelated_corpus = obtain_emission_probailities(list_of_words_in_question, "1_related_corpus.txt", "1_unrelated_corpus.txt")


##Parsing the question
questionWords_and_tags = pos_tag((word_tokenize((question))))
pprint(questionWords_and_tags)
refined_list_of_words_in_question = []
for w,t in questionWords_and_tags:
    if t == 'JJ' and e[(w,'K')] > e[(w,'NK')]:
        refined_list_of_words_in_question.append(w)
    if t == 'VBG' and e[(w,'K')] > e[(w,'NK')]:
        refined_list_of_words_in_question.append(w)
    elif not(t in ['WDT','JJ','WP','PRP$','WRB','VBP','VBN','IN','DT','TO', 'EX', 'VBZ', 'MD','RB']):
        refined_list_of_words_in_question.append(w)




for word in refined_list_of_words_in_question:
    refined_list_of_words_in_question.extend(obtain_synonyms(word))
 
tags =  viterbi(refined_list_of_words_in_question,q,e,initial_state_probabilities) 

query = [refined_list_of_words_in_question[idx] for idx in range(len(tags)) if tags[idx] == 'K']

print "For the question "
print question
print "words related to C programming are i.e. the query words are "

if is_the_query_relevent_LESK(set_of_words_related_corpus, set_of_words_unrelated_corpus, set(query) ):
    pprint(query)
else:
    print []
