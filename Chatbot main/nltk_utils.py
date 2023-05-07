import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
#nltk.download('punkt')
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    return stemmer.stem(word.lower())

words = ["program", "programs", "programmer", "programming", "programmers"]
b = [stem(w) for w in words]
print(b)
#print(tokenize("Hello how are you"))