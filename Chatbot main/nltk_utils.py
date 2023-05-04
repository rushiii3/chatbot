import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
#nltk.download('punkt')
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    return stemmer.stem(word)

words = ["program", "programs", "programmer", "programming", "programmers"]
print(stem(words))
#print(tokenize("Hello how are you"))