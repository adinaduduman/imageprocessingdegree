import requests
import json

api_url = "https://relatedwords.org/api/related?term="

def getRelatedWords(word, maxAmount):

    relatedWords = []
    response = requests.get(api_url + word, None)
    if response.status_code >= 200 & response.status_code < 400:
        terms = response.json()
        maxAmount = len(terms) if (len(terms) <= maxAmount) else maxAmount
        startIndex = 0
        while startIndex < maxAmount:
            relatedWords.append(terms[startIndex]["word"])
            startIndex = startIndex + 1
        return relatedWords
    else:
        print("Error while trying to call RelatedWords API")