'''Sample excecution command 
python LMwithKLDivergence.py corpus.txt test.txt'''

import nltk
import sys
import math
import string
import random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def kullbackLeibler(question,answer):
    #create a joint list
    jointList = question + answer
    #Set will avoid duplicates
    jointSet = set(jointList)
    #Generate frequency dist of test sentence
    questionFD = nltk.FreqDist(question)
    totalQuestionCount = len(question)
    totalAnswerCount = len(answer)
    #Generate frequency dist of answer
    ansFD = nltk.FreqDist(answer)
    jointFD = nltk.FreqDist(jointList)
    d = 0.0
    for ngram in jointSet:
        #Calculate the probability dist of specific ngram in answer sentence
        probAnsDist = float(ansFD[ngram])/totalAnswerCount
        #Calculate the joint probability dist of specific ngram
        jointProb = float(jointFD[ngram])/len(jointList)
        #Add smoothing to avoid zero division
        ansSmoothing = float(probAnsDist* 0.90) + float(0.10 * jointProb)
        #Calculate the probability dist of specific ngram in test sentence
        probQuesDist = float(questionFD[ngram])/totalQuestionCount
        logx = float(probQuesDist)/ansSmoothing
        if probQuesDist != 0:
            calculation = probQuesDist * math.log(logx)
            #Add the corresponding ngram entropy to overall distance measure
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
    kl_distance = {"entrophy":sys.maxint,"asnwer":""}
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
        entrophy = kullbackLeibler(test_bigrams,bigrams)
        if kl_distance['entrophy'] > entrophy:
            kl_distance['entrophy'] = entrophy
            kl_distance['sentence'] = review
        #print stats.entropy(test_bigrams,bigrams)
        
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
    fileObj.write("KL Entrophy Result: "+kl_distance['sentence'])
    if kl_distance['sentence'] == sentence:
        fileObj.write("KL Entrophy and Classifier results are same\n\n")
    else:
        fileObj.write("KL Entrophy and Classifier results are different\n\n")

    return result

#Unigram model for finding similar sentence in a big text file    
def unigram(sentenceList,test,fileObj,key):
    maxProb = 0
    resultLine = 1
    sentence = ""
    kl_distance = {"entrophy":sys.maxint,"asnwer":""}
    test_unigrams = test.strip().lower().split()
    for review in sentenceList:
        counts={}
        ans_unigrams =  review.strip().lower().split() 
        for word in ans_unigrams:
            counts[word] = counts.get(word, 0) + 1
            total = sum(counts.values()) + len(counts)
            prob = 1.0
        for w in test_unigrams:
            prob = prob * (float(counts.get(w, 0)+1)/total)
            if prob > maxProb:
                maxProb = prob
                sentence = review
                result = resultLine
        resultLine = resultLine + 1
        entrophy = kullbackLeibler(test_unigrams,ans_unigrams)
        if kl_distance['entrophy'] > entrophy:
            kl_distance['entrophy'] = entrophy
            kl_distance['sentence'] = review

    fileObj.write("Key: "+key+"\n")
    fileObj.write("Question: "+test+"\n")          
    fileObj.write("Result: "+sentence.strip()+"\n")
    fileObj.write("KL Entrophy Result: "+kl_distance['sentence'])
    if kl_distance['sentence'] == sentence:
        fileObj.write("KL Entrophy and Classifier results are same\n\n")
    else:
        fileObj.write("KL Entrophy and Classifier results are different\n\n")
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
        print "Secret Key:",unigramKey,"ACCURACY:",str(accuracyUnigram)
        print "Secret Key:",bigramKey,"ACCURACY:",str(accuracyBigram)
    else:
        print "Secret Key:",bigramKey,"ACCURACY:",str(accuracyBigram)
        print "Secret Key:",unigramKey,"ACCURACY:",str(accuracyUnigram)
    referenceFile = open("reference.txt",'w')
    referenceFile.write("Unigram: "+unigramKey+'\n')
    referenceFile.write("Bigram: "+bigramKey+'\n')


