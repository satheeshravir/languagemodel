# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:20:38 2015

@author: maxsteal
"""
import sys
import nltk
import copy
import math

def tokenizeSentenceLowerCase(sentence):
    tokens = [w.lower() for w in nltk.word_tokenize(sentence)]    
    return tokens


def questionsAnswersMap(corpusFilePath):
    corpusFile = open(corpusFilePath,"r")
    questionsMap = {}
    answersMap = {}
    for line in corpusFile:
        question,answer = line.split("\t")
        if question not in questionsMap:
            questionsMap[question] = [answer]        
        else:
            questionsMap[question].append(answer)
        if answer not in answersMap:
            answersMap[answer] = [question]
        else:
            answersMap[answer].append(question)
    return questionsMap,answersMap

def generateUnigramLMForString(string, shallowLM):
    unigramTokens = tokenizeSentenceLowerCase(string)
    totalTokensCount = len(unigramTokens)
    unigramFreqDist = nltk.FreqDist(unigramTokens)
    return generateLMFromFreqDist(unigramFreqDist,totalTokensCount, shallowLM)
    
def generateBigramLMForString(string, shallowLM):
    unigramTokens = tokenizeSentenceLowerCase(string)
    bigramTokens = list(nltk.bigrams(unigramTokens))
    totalTokensCount = len(bigramTokens)
    bigramFreqDist = nltk.FreqDist(bigramTokens)
    return generateLMFromFreqDist(bigramFreqDist, totalTokensCount, shallowLM)

def generateListOfTrainingStringsUnigramLM(totalTrainingStringsList, shallowLM):
    trainingStringLMList = []
    for string in totalTrainingStringsList:
        unigramLM = generateUnigramLMForString(string, shallowLM)
        trainingStringLMList.append(unigramLM)
    return trainingStringLMList
    
def generateOverallUnigramBackgroundLM(totalTrainingStringsList):
    tokens = []
    for string in totalTrainingStringsList:
        tokens += tokenizeSentenceLowerCase(string)
    overallFreqDist = nltk.FreqDist(tokens)
    return generateLMFreqDist(overallFreqDist,len(tokens))
    
def generateLMFreqDist(freqDist,tokensCount):
    lm = {}
    for token, count in freqDist.iteritems():
        lm[token] = float(count) / tokensCount
    return lm

def generateLMFromFreqDist(freqDist,tokensCount, shallowLM):
    lm = copy.deepcopy(shallowLM)
    for token, count in freqDist.iteritems():
        lm[token] = float(count) / tokensCount
    return lm

#Function that generates question model based on the equations  5, 2, 3, and 4 in papers
def generateQuestionModel(overallNgramBackgroundLM, trainingStringsLMList, queryTokens):
    ngrams = overallNgramBackgroundLM.keys()
    questionModel = {}
    smoothingFactor = 0.90
    for ngram in ngrams:
        numerator = 0.0
        denominator = 0.0
        for trainingStringLM in trainingStringsLMList:
            probOfNgramInString = smoothingFactor * trainingStringLM[ngram] + (1-smoothingFactor) * overallNgramBackgroundLM[ngram]
            probQueryTokenInString = 1.0
            for queryToken in queryTokens:
                if queryToken not in trainingStringLM:
                    probTokenInString = 0
                else:
                    probTokenInString = trainingStringLM[queryToken]
                if queryToken not in overallNgramBackgroundLM:
                    probTokenInBackgroundLM = 0
                else:
                    probTokenInBackgroundLM = overallNgramBackgroundLM[queryToken]
                probQueryTokenInString *= smoothingFactor * probTokenInString + (1-smoothingFactor) * probTokenInBackgroundLM
            numerator += probOfNgramInString * probQueryTokenInString
            denominator += probQueryTokenInString
        if denominator != 0:
            finalProb = numerator / denominator
        else:
            finalProb = 0.0
        questionModel[ngram] = finalProb
    return questionModel

def generateUnigramAnswersModel(answersList,shallowLM):
    answerModel = {}
    for answer in answersList:
        answerModel[answer] = generateUnigramLMForString(answer, shallowLM)
    return answerModel
                
def generateShallowLM(entireCorpus):
    corpusLM = {}
    for ngram,count in entireCorpus.iteritems():
        corpusLM[ngram] = 0
    return corpusLM        

def KLDivergence(answersModel, questionModel,overallNgramBackgroundLM):
    smoothing = 0.90
    ngrams = overallNgramBackgroundLM.keys()
    minDistance = sys.maxint
    d = 0
    for answer, answerModel in answersModel.iteritems():
        for ngram in ngrams:
            probAnsDist = answerModel[ngram]
            jointProbDist = overallNgramBackgroundLM[ngram]
            ansSmoothing = float(probAnsDist* smoothing) + float((1-smoothing) * jointProbDist)
            probQuesDist = questionModel[ngram]
            logx = float(probQuesDist)/ansSmoothing
            if probQuesDist != 0:
                calculation = probQuesDist * math.log(logx)
                #Add the corresponding ngram entropy to overall distance measure
                d = d + calculation
        if d < minDistance:
            minDistance = d
            finalAnswer = answer
    return finalAnswer
        
            
    

if __name__ == "__main__":
    debug = 1
    #Input file
    corpusFilePath = sys.argv[1]
    
    questionsMap, answersMap = questionsAnswersMap(corpusFilePath)
    totalQuestionsList = questionsMap.keys()
    totalAnswersList = answersMap.keys()
    
    #Training set S
    totalTrainingStringsList = totalQuestionsList + totalAnswersList    
    #Overall baclground LM with all the training strings for smoothing
    overallUnigramBackgroundLM = generateOverallUnigramBackgroundLM(totalTrainingStringsList)
    
    shallowUnigramLM = generateShallowLM(overallUnigramBackgroundLM)
    #LM for the trainint string set S
    trainingStringsUnigramLMList = generateListOfTrainingStringsUnigramLM(totalTrainingStringsList,shallowUnigramLM)
    
    #Sample Query
    query = "Can I ask you some query?"
    queryUnigramTokens = tokenizeSentenceLowerCase(query)
    #Question LM p(w|Q)
    questionModel = generateQuestionModel(overallUnigramBackgroundLM, trainingStringsUnigramLMList, queryUnigramTokens)
    #Answer Language Model
    answersModel = generateUnigramAnswersModel(totalAnswersList,shallowUnigramLM)
    #Rank the pseudo answer
    pseudoAnswer = KLDivergence(answersModel, questionModel,overallUnigramBackgroundLM)
    
    print pseudoAnswer
    
    
    
    