import numpy as np
import math
import os

def crTokens(emails):
    global hamWords
    global spamWords
    global spamCount
    global hamCount
    global hamProb, spamProb
    global hamTrains, spamTrains
    hamTrains = [line for line in emails if 'ham' in line]
    spamTrains = [line for line in emails if 'spam' in line]
    spamWords = {}
    spamCount = 0
    for email in spamTrains:
        f = open("emails/%s" % email, errors='replace')
        for line in f:
            words = line.split()
            for word in words:
                if word not in stopwords and word not in \
                        ['.', ',', ';', '?', '!', ':', 'Subject:', ')', '(', '[', ']', '{', '}'] and not word.isdigit():        # irasjel
                    spamCount += 1
                    if word.lower() in spamWords:
                        spamWords[word.lower()] += 1
                    else:
                        spamWords[word.lower()] = 1
    hamWords = {}
    hamCount = 0
    for email in hamTrains:
        f = open("emails/%s" % email, errors='replace')
        for line in f:
            words = line.split()
            for word in words:
                if word not in stopwords and word not in \
                        ['.', ',', ';', '?', '!', ':', 'Subject:', ')', '(', '[', ']', '{', '}'] and not word.isdigit():
                    hamCount += 1
                    if word.lower() in hamWords:
                        hamWords[word.lower()] += 1
                    else:
                        hamWords[word.lower()] = 1
    hamProb = len(hamTrains)/len(emails)
    spamProb = len(spamTrains) / len(emails)

def wordProbs():
    global hamWords, spamWords
    allWords = {}
    for word, count in hamWords.items():
        WHP = count / hamCount
        if word in spamWords:
            WSP = spamWords[word] / spamCount
        else:
            WSP = 0.00000001
        allWords[word] = (WHP, WSP)
    for word, c in spamWords.items():
        if word not in hamWords:
            WSP = c / spamCount
            WHP = 0.00000001
            allWords[word] = (WHP, WSP)
    return allWords

def additiveSmoothing(alpha, dLen):
    auxDict = {}
    for word, count in hamWords.items():
        hamPr = (count + alpha) / (alpha * dLen + hamCount)
        if word in spamWords:
            spamPr = (spamWords[word] + alpha) / (alpha * dLen + spamCount)
        else:
            spamPr = alpha / (alpha * dLen + spamCount)
        auxDict[word] = (hamPr, spamPr)
    for word, c in spamWords.items():
        if word not in hamWords:
            spamPr = (c + alpha) / (alpha * dLen + spamCount)
            hamPr = alpha / (alpha * dLen + hamCount)
            auxDict[word] = (hamPr, spamPr)
    return auxDict

def naiveBayes(email, probDict, learns):
    global hamProb, spamProb
    if learns:
        f = open(email, errors='replace')
    else:
        f = open("emails/%s" % email, errors='replace')
    auxDict = {}
    for line in f:
        for word in line.split():
            if word not in stopwords and word not in \
                    ['.', ',', ';', '?', '!', ':', 'Subject:', ')', '(', '[', ']', '{', '}'] and not word.isdigit():
                if word.lower() in auxDict.keys():
                    auxDict[word.lower()] += 1
                else:
                    auxDict[word.lower()] = 1
    r = 0
    for word, count in auxDict.items():
        if word in probDict.keys():
            r += count * (math.log(probDict[word][1]) - math.log(probDict[word][0]))
    if hamProb == 0:
        r += math.log(spamProb)
    else:
        if spamProb == 0:
            r += -math.log(hamProb)
        else:
            r += math.log(spamProb) - math.log(hamProb)
    return r

def accuracy(emails, probDict, learns):
    allMails = 0
    ok = 0
    for email in emails:
        allMails += 1
        if naiveBayes(email, probDict, learns) > 0 and 'spam' in email:
            ok += 1
        else:
            if 'ham' in email:
                ok += 1
    return ok / allMails        # helyesseg

def crossValidation(allMail):
    global hamCount, spamCount
    global hamWords, spamWords
    global hamProb, spamProb
    probDict = {}
    temp = {}
    alphas = [0.1,  0.05, 0.01, 0.2, 0.5]
    parts = 5
    partLen = int(len(allMail) / parts)
    corrects = [0] * len(alphas)
    for alpha in alphas:
        for i in range(parts):
            if i == 0:
                train = allMail[partLen+1 : parts * partLen]
            elif i == 4:
                train = allMail[0 : (parts-1)*partLen]
            else:
                v = parts-i-1
                k = i-1
                train = allMail[0 : k*partLen]
                train += allMail[v*partLen : parts*partLen]
            crTokens(train)
            temp = wordProbs()
            probDict = additiveSmoothing(alpha, len(temp))
            corrects[i] += accuracy(allMail[i * partLen : (i+1) * partLen], probDict, False)

    corrects = list(map(lambda p: (1-p/len(alphas))*100, corrects))
    print(corrects)
    return alphas[corrects.index(min(corrects))], min(corrects)


def addHam(mail):
    global hamWords
    f = open(mail, errors='replace')
    for line in f:
        for word in line.split():
            if word not in stopwords and word not in \
                    ['.', ',', ';', '?', '!', ':', 'Subject:', ')', '(', '[', ']', '{', '}'] and not word.isdigit():
                if word.lower() in hamWords.keys():
                    hamWords[word.lower()] += 1
                else:
                    hamWords[word.lower()] = 1


def addSpam(mail):
    global spamWords
    f = open(mail, errors='replace')
    for line in f:
        for word in line.split():
            if word not in stopwords and word not in \
                    ['.', ',', ';', '?', '!', ':', 'Subject:', ')', '(', '[', ']', '{', '}'] and not word.isdigit():
                if word.lower() in spamWords.keys():
                    spamWords[word.lower()] += 1
                else:
                    spamWords[word.lower()] = 1


def bayes(mails):
    global hamWords, spamWords
    global hamCount, spamCount
    global allDict
    counter = 0
    learnedHam = []
    learnedSpam = []
    while True:
        for mail in mails:
            value = naiveBayes(mail, allDict, True)
            if value > 5:
                learnedHam.append(mail)
                mails.remove(mail)
            elif value < -5:
                learnedSpam.append(mail)
                mails.remove(mail)
        if learnedHam == [] and learnedHam == []:
            break
        counter += len(learnedHam) + len(learnedSpam)
        for mail in learnedHam:
            addHam(mail)
        hamCount = sum(hamWords.values())
        for mail in learnedSpam:
            addSpam(mail)
        spamCount = sum(spamWords.values())
        learnedHam = []
        learnedSpam = []
    print('Learned from', counter, 'mails.')

file = open("stopwords2.txt", "r")
stopwords = [line.rstrip('\n') for line in file]
file = open("train.txt", "r")
hamTrains = [line.rstrip('\n') for line in file if 'ham' in line]
file = open("train.txt", "r")
spamTrains = [line.rstrip('\n') for line in file if 'spam' in line]
file = open('train.txt', "r")
trainMails = [line.rstrip('\n') for line in file]
file = open('test.txt', "r")
testMails = [line.rstrip('\n') for line in file]

spamWords = {}
spamCount = 0
hamWords = {}
hamCount = 0

crTokens(trainMails)
hamProb = len(hamTrains)/(len(hamTrains) + len(spamTrains))
spamProb = len(spamTrains)/(len(hamTrains) + len(spamTrains))
allWords = wordProbs()
print("Naive Bayes:")
print((1 - accuracy(trainMails, allWords, False)) * 100)
print((1 - accuracy(testMails, allWords, False)) * 100)

additiveDict = additiveSmoothing(0.1, len(allWords))
print("Additive smoothing\nalpha = 0.1")
print((1 - accuracy(trainMails, additiveDict, False)) * 100)
print((1 - accuracy(testMails, additiveDict, False)) * 100)

print("Additive smoothing\nalpha = 0.01")
additiveDict = additiveSmoothing(0.01, len(allWords))
print((1 - accuracy(trainMails, additiveDict, False)) * 100)
print((1 - accuracy(testMails, additiveDict, False)) * 100)

print("Additive smoothing\nalpha = 1")
additiveDict = additiveSmoothing(1, len(allWords))
print((1 - accuracy(trainMails, additiveDict, False)) * 100)
print((1 - accuracy(testMails, additiveDict, False)) * 100)

print('Cross validation:')
print(crossValidation(testMails))

crTokens(trainMails)
allDict = wordProbs()
mailsToLearn = list(map(lambda p: 'ssl/' + p, os.listdir('ssl')))
print('Bayes learning:')
bayes(mailsToLearn)
allWords = wordProbs()
print((1 - accuracy(testMails, allWords, False)) * 100)