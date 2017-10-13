# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 20:19:01 2017

@author: shubra
"""

from keras.preprocessing.text import Tokenizer
nb_words = 3
tokenizer = Tokenizer(num_words=nb_words)
tokenizer.fit_on_texts(["The sun is shining in  June!","September is grey.","Life is beautiful in August.","I like it","This and other things?"])
"""
Index of Each word
"""
print(tokenizer.word_index)

""" PRINTS FOLLOWING
{'is': 1, 'in': 2, 'the': 3, 'sun': 4, 'shining': 5, 'june': 6, 'september': 7,
 'grey': 8, 'life': 9, 'beautiful': 10, 'august': 11, 'i': 12, 'like': 13, 
 'it': 14, 'this': 15, 'and': 16, 'other': 17, 'things': 18}
"""



"""
You need to read this as: take only words with an index less or equal to 3 (the constructor parameter)

Prints [[1]] as only "is" is the word that has Index less or equal to 3
"""

print("\n",tokenizer.texts_to_sequences(["June is beautiful and I like it!"]))




"""
Parameter less tokenizer it will read all words
"""
tokenizer = Tokenizer()
texts = ["The sun is shining in June!","September is grey.","Life is beautiful in August.","I like it","This and other things?"]
tokenizer.fit_on_texts(texts)
print("\n",tokenizer.word_index)

"""
{'is': 1, 'in': 2, 'the': 3, 'sun': 4, 'shining': 5, 'june': 6, 'september': 7,
 'grey': 8, 'life': 9, 'beautiful': 10, 'august': 11, 'i': 12, 'like': 13, 
 'it': 14, 'this': 15, 'and': 16, 'other': 17, 'things': 18}
"""
print("\n",tokenizer.texts_to_sequences(["June is beautiful and I like it!"]))
""" PRINTS FOLLOWING
[[6, 1, 10, 16, 12, 13, 14]]
"""
print("\n",tokenizer.word_counts)

""" PRINTS FOLLOWING
 OrderedDict([('the', 1), ('sun', 1), ('is', 3), ('shining', 1), ('in', 2),
 ('june', 1), ('september', 1), ('grey', 1), ('life', 1), ('beautiful', 1),
 ('august', 1), ('i', 1), ('like', 1), ('it', 1), ('this', 1), ('and', 1), 
 ('other', 1), ('things', 1)])
"""

print("\n","Was lower-case applied to %s sentences?: %s"%(tokenizer.document_count,tokenizer.lower))
""" PRINTS FOLLOWING
Was lower-case applied to 5 sentences?: True
there are 5 sentences in the document each " "," makes one sentence
"""

"""
"The sun is shining in  June!",
"September is grey.",
"Life is beautiful in August.",
"I like it","This and other things?"
"""
import numpy as np
np.set_printoptions(threshold=np.nan) # use this to print full numpy array

y =tokenizer.texts_to_matrix(["June is beautiful and I like it!"])
print("\n",y)

""" PRINTS THE FOLLOWING
 [[ 0.  1.  0.  0.  0.  0.  1.  0.  0.  0.  1.  0.  1.  1.  1.  0.  1.  0.
   0.]]
 
  [[ 0.  1=is.  0.  0.  0.  0.  1=june(index6).  0.  0.  0.  1=beautiful(index10)
  .  0.  1=i=index12.  1=like=index13.  1=it=index14.  0.  1=and=index16.  0.    0.]]
"""
print("\n",tokenizer.texts_to_matrix(["June is beautiful and I like it!","Like August"]))

""" PRINTS 2 ROWS for 2 Sentence THE FOLLOWING
 [[ 0.  1.  0.  0.  0.  0.  1.  0.  0.  0.  1.  0.  1.  1.  1.  0.  1.  0.
   0.]]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  1.  0.  0.  0.  0.
   0.]]
 
  [[ 0.  1=is.  0.  0.  0.  0.  1=june(index6).  0.  0.  0.  1=beautiful(index10)
  .  0.  1=i=index12.  1=like=index13.  1=it=index14.  0.  1=and=index16.  0.    0.]]
  
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1=august=index11.  0.  1=like=index13. 
 0.  0.  0.  0.    0.]] 
"""

