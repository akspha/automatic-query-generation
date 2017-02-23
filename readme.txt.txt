Libraries:
NLTK  set of libraries has been used for certain utilities and can be installed from the following website:
https://pypi.python.org/pypi/nltk

Including corpora:
Related corpus for this project is embraced by the file: 1_related_corpus.txt
Unrelated corpus for this project is embraced by the file:1_unrelated_corpus.txt
To use other corpora, change the filenames by altering the following variables in enhanced_query_generation.py 
related_corpus 
unrelated_corpus

How to run the programs:
To run the basline, open generate_query_baseline.py inside IDLE and press F5
then on the REPL, call generate_query_baseline() with any question as a string input.

To run the improved program, open enhanced_query_generation.py inside IDLE and press F5
then on the REPL, enter a question where a prompt awaits your question

(Run enhanced_query_generation_with_synsets.py similarly. This program is very slow due to exhaustive enumeration
over synonyms and is being submitted to show the effect of using synsets. Corpora can be changed by changing arguments to obtain_emission_probailities() )