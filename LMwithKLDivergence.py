'''Sample excecution command
python languageModel.py /home/maxsteal/Downloads/labeledTrainData.tsv'''

import nltk
import sys
import math
import string
import random

#Generates random secret key
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

#Caculates Kullback-Leiber measure given the ngram (unigram, bigram) list of question and answer
def kullbackLeibler(question,answer):
    #Frequency distribution of question
    questionFD = nltk.FreqDist(question)
    #Frequency distribution of answer
    ansFD = nltk.FreqDist(answer)
    #KL distance measure
    d = 0
    #For each ngram in the answer 
    for ans in answer:
        prob_ans = float(ansFD[ans])/len(answer)
        prob_test = float((questionFD[ans]+1))/(len(answer)+len(question))
        logx = prob_test/prob_ans
        calculation = prob_test * math.log(logx)
        d = d + calculation
    return d

        

#Bigram model for finding similar sentence in a big text file
def bigram(sentenceList,test,fileObj,key):
    #Tokenize the test sentence
    test_words = nltk.word_tokenize(test.strip())
    test_tokens = [test_token.lower() for test_token in test_words]
    #Form a bigram list of test sentence
    test_bigrams = list(nltk.bigrams(test_tokens))
    maxProb = 0.0
    sentence = ""
    resultLine = 1
    result = 0
    #Iterate through each line in the text file
    for review in sentenceList:
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
            result = resultLine
        resultLine = resultLine + 1
        
    fileObj.write("Key: "+key+"\n")
    fileObj.write("Question: "+test+"\n")          
    fileObj.write("Result: "+sentence.strip()+"\n")
    answer_tokens = [answer_token.lower() for answer_token in sentence.strip().split()]
    answer_bigrams = list(nltk.bigrams(answer_tokens))
    fileObj.write("Entropy: "+ str(kullbackLeibler(test_bigrams,answer_bigrams))+'\n\n')
    return result

#Unigram model for finding similar sentence in a big text file    
def unigram(sentenceList,test,fileObj,key):
    maxProb = 0
    resultLine = 1
    sentence = ""
    for review in sentenceList:
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
                result = resultLine
        resultLine = resultLine + 1
    fileObj.write("Key: "+key+"\n")
    fileObj.write("Question: "+test+"\n")          
    fileObj.write("Result: "+sentence.strip()+"\n")
    answer_tokens = [answer_token.lower() for answer_token in sentence.strip().split()]
    fileObj.write("Entropy: "+  str(kullbackLeibler(test.strip().lower().split(),answer_tokens))+'\n\n')
    return result

if __name__ == "__main__":
    debug = 1
    #Input file
    filePath = sys.argv[1]
    testPath = sys.argv[2]
    #train = pd.read_csv(filePath, header=0, delimiter="\t", quoting=3)
    fileObj = open(filePath)
    sentenceList = fileObj.readlines()
    unigramKey = id_generator()
    bigramKey = id_generator()
    resultFile = open("result.txt",'w')
    accuracyUnigram = 0
    accuracyBigram = 0
    inputData = open(testPath).readlines()
    for data in inputData:
        test = data.split(":::")
        if random.randrange(1,101) % 2:
            bigramAns = bigram(sentenceList,test[0],resultFile,bigramKey)
            unigramAns = unigram(sentenceList,test[0],resultFile,unigramKey)
        else:
            unigramAns = unigram(sentenceList,test[0],resultFile,unigramKey)            
            bigramAns = bigram(sentenceList,test[0],resultFile,bigramKey)
        if unigramAns == int(test[1].strip()):
            accuracyUnigram+=1
        if bigramAns == int(test[1].strip()):
            accuracyBigram+=1
    if accuracyUnigram != 0:
        accuracyUnigram = float(accuracyUnigram)/len(inputData)
    if accuracyBigram != 0:
        accuracyBigram = float(accuracyBigram)/len(inputData)
    if random.randrange(1,101) % 2:
        print "Key",unigramKey,"ACCURACY",str(accuracyUnigram)
        print "Key",bigramKey,"ACCURACY",str(accuracyBigram)
    else:
        print "Key",bigramKey,"ACCURACY",str(accuracyBigram)
        print "Key",unigramKey,"ACCURACY",str(accuracyUnigram)



