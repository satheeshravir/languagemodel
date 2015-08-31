'''Sample excecution command
python languageModel.py /home/maxsteal/Downloads/labeledTrainData.tsv'''

import pandas as pd  
import nltk
import sys

#Bigram model for finding similar sentence in a big text file
def bigram(train,test):
    #Tokenize the test sentence
    test_words = nltk.word_tokenize(test.strip())
    test_tokens = [test_token.lower() for test_token in test_words]
    #Form a bigram list of test sentence
    test_bigrams = list(nltk.bigrams(test_tokens))
    maxProb = 0.0
    sentence = ""
    #Iterate through each line in the text file
    for review in train['review']:
        #Tokenize each sentence
        words = review.strip().split()    
        tokens = [token.lower() for token in words]
        #Find the frequency of words in the sentence
        fd = nltk.FreqDist(tokens)
        #Form bigrams of the sentence
        bigrams = list(nltk.bigrams(tokens))
        bigramsFD = nltk.FreqDist(bigrams)
        prob = 1.0
        #Calculate the conditional probability of each bigram in the test sentence
        for test_bigram in test_bigrams:
            prob = prob * (bigramsFD[test_bigram]+1)/(fd[test_bigram[0]]+len(test_bigrams))
        #Store the max probability and similar sentence
        if maxProb < prob:
            maxProb = prob
            sentence = review
    print "*****************************************"
    print "Bigram result for similar sentence"
    print "*****************************************"
    print sentence

#Unigram model for finding similar sentence in a big text file    
def unigram(train,test):
    maxProb = 0
    sentence = ""
    for review in train['review']:
        counts={}
        for word in review.strip().lower().split():
            counts[word] = counts.get(word, 0) + 1
            total = sum(counts.values()) + len(counts)
            prob = 1.0
        for w in test.strip().lower().split():
            prob = prob * (float(counts.get(w, 0)+1)/total)
            if prob > maxProb:
                maxProb = prob
                sentence = review
    print "*****************************************"
    print "Unigram result for similar sentence"
    print "*****************************************"
    print sentence


if __name__ == "__main__":
    debug = 1
    #Input file
    filePath = sys.argv[1]
    train = pd.read_csv(filePath, header=0, delimiter="\t", quoting=3)
    test = raw_input("Enter the test sentence:")
    bigram(train,test)
    unigram(train,test)
    # Store and process command line
    


