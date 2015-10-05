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

def generateModelUsingEquation4(overallNgramBackgroundLM, probDist, smoothing):
    ngrams = overallNgramBackgroundLM.keys()
    answerModel = {}
    for ngram in ngrams:
        answerModel[ngram] = smoothing * probDist[ngram] + (1-smoothing) * overallNgramBackgroundLM[ngram]
    return answerModel

#Function that generates question model based on the equation 2 
def generateQuestionModelUsingEquation2(overallNgramBackgroundLM, trainingStringsLMList, queryTokens, smoothingFactor):
    ngrams = overallNgramBackgroundLM.keys()
    questionModel = {}
    for ngram in ngrams:
        numerator = 0.0
        denominator = 0.0
        for trainingStringLM in trainingStringsLMList:
            probOfNgramInString = smoothingFactor * trainingStringLM[ngram] + (1-smoothingFactor) * overallNgramBackgroundLM[ngram]
            probQueryTokenInString = 1.0
            for queryToken in queryTokens:
                if queryToken in overallNgramBackgroundLM:
                    if queryToken not in trainingStringLM:
                        probTokenInString = 0
                    else:
                        probTokenInString = trainingStringLM[queryToken]
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
    

#Generate model based on equation 4
def generateNgramModelsFromMapUsingEquation4(stringMap,shallowLM,overallNgramBackgroundLM, smoothing):
    answerModel = {}
    for key, string in stringMap.iteritems():
        ngramLM = generateUnigramLMForString(string, shallowLM)
        answerModel[key] = generateModelUsingEquation4(overallNgramBackgroundLM,ngramLM,smoothing)
    return answerModel
 
#Generate model based on equation 4
def generateNgramModelsFromListUsingEquation4(stringList,shallowLM,overallNgramBackgroundLM, smoothing):
    answerModel = {}
    for string in stringList:
        ngramLM = generateUnigramLMForString(string, shallowLM)
        answerModel[string] = generateModelUsingEquation4(overallNgramBackgroundLM,ngramLM,smoothing)
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
    finalAnswer = ""
    for answer, answerModel in answersModel.iteritems():
        d = -1
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
        #print d,answer
        if d < minDistance:
            minDistance = d
            finalAnswer = answer
    return finalAnswer

def generatePseudoAnswersMap(answersMap):
    pseudoAnswersMap = {}
    for answer, questionsList in answersMap.iteritems():
        pseudoAnswersMap[answer] = " ".join(questionsList)
    return pseudoAnswersMap
   
def generateOverallQuesAnsPairsLM(questionsMap):
    totalStrings = []
    for question, answerList in questionsMap.iteritems():
        totalStrings += [question]+answerList
    overallAnsQuesPairsBackgroundLM = generateOverallUnigramBackgroundLM(totalStrings)
    return overallAnsQuesPairsBackgroundLM

def generateQuestionAnswerPairModelUsingEquation4(questionsMap, overallAnsQuesPairsBackgroundLM):
    questionAnswerPairModel = {}
    shallowLM = generateShallowLM(overallAnsQuesPairsBackgroundLM)
    for question, answerList in questionsMap.iteritems():
        questionProbDist = generateUnigramLMForString(question, shallowLM)
        
        questionModel = generateModelUsingEquation4(overallAnsQuesPairsBackgroundLM,questionProbDist, smoothing)
        for answer in answerList:            
            ansProbDist = generateUnigramLMForString(answer, shallowLM)
            ansModel = generateModelUsingEquation4(overallAnsQuesPairsBackgroundLM, ansProbDist, smoothing)
            questionAnswerPairModel[question+answer] = [questionModel, ansModel]
    return questionAnswerPairModel

def generateQuestionModelUsingEquation6(questionsMap, queryUnigramTokens, overallUnigramAnswersBackgroundLM, smoothing):
    finalQuestionModel = {}
    overallAnsQuesPairsBackgroundLM = generateOverallQuesAnsPairsLM(questionsMap)
    questionAnswerPairModel = generateQuestionAnswerPairModelUsingEquation4(questionsMap, overallAnsQuesPairsBackgroundLM)
    for ngram in overallUnigramAnswersBackgroundLM.keys():
        numerator = 0.0
        denominator = 0.0
        for key, questionAnswerModel in questionAnswerPairModel.iteritems():
            answerModel = questionAnswerModel[1]
            questionModel = questionAnswerModel[0]
            ngramInAnswerModel = answerModel[ngram]
            queryInQuestionModel = 1.0
            for query in queryUnigramTokens:
                if query in questionModel:
                    queryInQuestionModel *= questionModel[query]
            numerator += (ngramInAnswerModel * queryInQuestionModel)
            denominator += queryInQuestionModel
        if denominator != 0:
            finalProb = numerator / denominator
        else:
            finalProb = 0
        finalQuestionModel[ngram] = finalProb
    return finalQuestionModel
            
            
if __name__ == "__main__":
    debug = 1
    #Corpus File
    corpusFilePath = sys.argv[1]
    #Test File
    testFilePath = sys.argv[2]
    smoothing = 0.90
    questionsMap, answersMap = questionsAnswersMap(corpusFilePath)
    totalQuestionsList = questionsMap.keys()
    totalAnswersList = answersMap.keys()
    
    #Background model generated for smoothing
    overallUnigramQuestionsBackgroundLM = generateOverallUnigramBackgroundLM(totalQuestionsList)
    overallUnigramAnswersBackgroundLM = generateOverallUnigramBackgroundLM(totalAnswersList)
    #skeleton for LMs
    shallowQuestionBackgroundLM = generateShallowLM(overallUnigramQuestionsBackgroundLM)
    shallowAnswerBackgroundLM = generateShallowLM(overallUnigramAnswersBackgroundLM)

    #Generate pseudo answers map
    pseudoAnswersMap = generatePseudoAnswersMap(answersMap)
    #Pseudo Answers LM created using equation 4    
    individualPseudoAnswersLM = generateNgramModelsFromMapUsingEquation4(pseudoAnswersMap, shallowQuestionBackgroundLM, overallUnigramQuestionsBackgroundLM, smoothing)

    #Answers and Question LM created using equation 4    
    individualAnswersLM = generateNgramModelsFromListUsingEquation4(totalAnswersList,shallowAnswerBackgroundLM, overallUnigramAnswersBackgroundLM, smoothing)
    testFile = open(testFilePath,"r")
    for line in testFile:
        #Sample Query
        query = line
        queryUnigramTokens = tokenizeSentenceLowerCase(query)
        print "+++++++++++++++++++++++++++++"
        print "Question Model (Section 3.4)"
        print "+++++++++++++++++++++++++++++"
        print "Question:",line
        
        #Model generated using the equation 2        
        questionModel = generateQuestionModelUsingEquation2(overallUnigramQuestionsBackgroundLM,individualPseudoAnswersLM.values(),queryUnigramTokens, smoothing)
        print "Answer:",KLDivergence(individualPseudoAnswersLM, questionModel, overallUnigramQuestionsBackgroundLM)    
        print "+++++++++++++++++++++++++++++"
        print "Answer Model (Section 3.5)"
        print "+++++++++++++++++++++++++++++"
        print "Question:",line
        #Model generated using the equation 6        
        questionModel = generateQuestionModelUsingEquation6(questionsMap, queryUnigramTokens, overallUnigramAnswersBackgroundLM, smoothing)
        print "Answer:",KLDivergence(individualAnswersLM, questionModel, overallUnigramAnswersBackgroundLM)    

        
    testFile.close()
    
    
    
    