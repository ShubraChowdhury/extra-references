# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 08:12:22 2017

@author: shubra
"""

import re

def count_words(text):
    """Count how many times each unique word occurs in text."""
    counts = dict()  # dictionary of { <word>: <count> } pairs to return
    #counts={'guests':1}
    # TODO: Convert to lowercase
    text = text.lower()
    #print(text)
    
    # TODO: Split text into tokens (words), leaving out punctuation
    # (Hint: Use regex to split on non-alphanumeric characters)
    text = re.sub(r'[^\w\s]','',text)
    #words=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', text)
    words = text.split()
    #for x,y in enumerate(words):
    for i in words:
     #print(x, '\t',y)
     if i in counts:
        counts[i] +=1
     else:
         counts[i] =1
     #counts= {y:1}
     
    
    
    # TODO: Aggregate word counts using a dictionary
    #print(counts)
    return counts


def test_run():
    with open("input.txt", "r") as f:
        text = f.read()
        counts = count_words(text)
        sorted_counts = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
        
        print("10 most common words:\nWord\tCount")
        for word, count in sorted_counts[:10]:
            print("{}\t{}".format(word, count))
        
        print("\n10 least common words:\nWord\tCount")
        for word, count in sorted_counts[-10:]:
            print("{}\t{}".format(word, count))


if __name__ == "__main__":
    test_run()
