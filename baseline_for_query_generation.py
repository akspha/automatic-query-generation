############################################################################################################
##Akshay Surendra Phadnis
##16 November
##NLP Project: Query generation
##baseline_for_query_generation.py implements a naive algorithm to obtain queries from a question
##so that, they can be used for Question-Answering by searching for answers
##in corpora of unstructured text
############################################################################################################
def generate_query_baseline(question):
    """
    Input: Question for the QA system (a string)
    
    Purpose: Takes a question and returns a collection words that can be fired against unstructured
     text to obtain answer to the question

    Output: a set of strings that represents a collection of query words
    
    """
    ##This naive algorithm is going to randomly select some words from the question and throw them out as
    ##the output

    ##Importing a function that allows choosing a random element from a list
    from random import sample as select_k_random_elements_from_list
    from random import uniform
   

    ##Convert all words in question to lower case and remove trailing question mark.
    ##After that, split the string into a list of words
    question_words = question.lower().strip('?').split()

    ##return a set of randomly chosen words.
    ##Note: how many elements to choose is also resolved randomly i.e. we generate a random number and
    ## choose those many elements. The second argument to uniform is len(question_words) because
    # the number of query words generated cannot be more then the number of words in the question itself
    return  set( select_k_random_elements_from_list( question_words, int(uniform(1, len(question_words) ) )  )    )


###Testing generate_query

##>>> generate_query_baseline("How are you today, Sir ?")
##set(['today,', 'are'])
##>>> generate_query_baseline("How are you today, Sir ?")
##set(['sir', 'you', 'today,', 'are'])
##>>> generate_query_baseline("How are you today, Sir ?")
##set(['sir', 'today,', 'are'])


##>>> generate_query_baseline("How to roast chicken ?")
##set(['to', 'roast'])
##>>> generate_query_baseline("How to roast chicken ?")
##set(['to', 'chicken', 'how'])
##>>> generate_query_baseline("How to roast chicken ?")
##set(['chicken', 'roast'])
##>>> generate_query_baseline("How to roast chicken ?")
##set(['how', 'to'])


##Drawbacks of this naive algorithm:

## It may generate a list of unimportant words like articles, interjections, etc. For instance, a list [the, how, are]
 ##barely seems fitting the purpose of generating a good query
    ## solution: Viterbi algorithm (HMM)

##The number of words to be generated is not fixed
    ## solution: solution to above problem will handle this

##Even though the returned query cannot be empty, it can be awfully small like one word!
    ## solution:  Viterbi will tackle this



## The relevance of the query words generated to the corpus of interest is not being assessed
    ## solution: WSD

## The relevance of a question to the related domain is not being assessed..
    ##proposed solution: WSD as in is_the_question_relevent_LESK() as defined in enhanced_query_generation.py 

##People are often intersted in how? questions. Insuch cases, the answers should pertain to the main verb in the question.
##The importance of this main verb is not being assessed.
    ## solution: retaining keywords that belong to VB part of speech
