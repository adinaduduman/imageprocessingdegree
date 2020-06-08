import os
import numpy as np
import re
from nltk.corpus import stopwords
import scipy

from ComplainSorter.src.word_proc_alg.runAlgorithm import runAlgorithm
from ComplainSorter.src.preprocessing.preprocessing import run_preproc
from ComplainSorter.src.cleanScript import runScript
from ComplainSorter.services.relatedWordsApi import getRelatedWords


def preprocessing():
    runScript()
    directory = '../data/1_raw_complains'
    files = os.listdir(directory)
    for (i, f) in enumerate(files):
        run_preproc(f)

    runAlgorithm()
    print("Pre processing complete!")


def buildRelatedWordList(directoryName, isCategory):
    directory = '../data/' + directoryName
    files = os.listdir(directory)

    for (i, f) in enumerate(files):
        relatedWords = []
        with open(directory + "/" + f, 'r') as file:
            for line in file:
                for word in line.split():
                    relatedWords.append(word)
                    words = getRelatedWords(word, 2)
                    for item in words:
                        relatedWords.append(item)
        file.close()
        if isCategory:
            with open(directory + "/" + f, 'w') as file:
                for relatedWord in relatedWords:
                    file.write(" " + relatedWord)
            file.close()
        else:
            newFilePath = f'../data/6_lexical_semantics/{f}'
            if not os.path.exists(newFilePath):
                with open(newFilePath, 'w') as newFile:
                    for relatedWord in relatedWords:
                        newFile.write(" " + relatedWord)
                newFile.close()

    print("Building the lexical semantics complete! ")


def postprocessing():
    buildRelatedWordList("5_complains_text", False)

    # build category dictionary
    categoryDictionary = {}
    catDirectory = '../data/sorting_categories'
    files = os.listdir(catDirectory)

    for (i, f) in enumerate(files):
        categoryWords = ""
        with open(catDirectory + "/" + f, 'r') as file:
            for line in file:
                for word in line.split():
                    categoryWords = categoryWords + " " + word
        categoryDictionary[f.split('.')[0]] = categoryWords

    semanticsDirectory = '../data/6_lexical_semantics'
    files = os.listdir(semanticsDirectory)

    for (i, f) in enumerate(files):
        semanticsText = ""
        resultDictionary = {}
        with open(semanticsDirectory + "/" + f, 'r') as file:
            for line in file:
                for word in line.split():
                    semanticsText = semanticsText + " " + word
        for categoryName in categoryDictionary:
            categoryText = categoryDictionary[categoryName]
            similarityPercentage = cosine_distance_wordembedding_method(semanticsText, categoryText)
            resultDictionary[categoryName] = similarityPercentage
        print(resultDictionary)


def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    with open(gloveFile, encoding="utf8" ) as f:
        content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model


def preprocessModelWords(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    cleaned_words = list(set([w for w in words if w not in stopword_set]))

    return cleaned_words


def cosine_distance_wordembedding_method(s1, s2):
    vector_1 = procVector(s1)
    vector_2 = procVector(s2)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    return round((1-cosine)*100,2)


def procVector(s):
    vector_1 = []
    for word in preprocessModelWords(s):
        try:
            vector_1.append(model[word])
        except KeyError:
            continue
    return np.mean(vector_1, axis=0)


if __name__ == '__main__':
    preprocessing()
    model = loadGloveModel("postprocessing/glove.6B.50d.txt")
    postprocessing()

