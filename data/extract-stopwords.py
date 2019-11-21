import string
import json

FILE_DIR = "./"
STOPWORDS = "stopwords.txt"
DOCUMENT_VOCAB = "document.vocab"
OUTPUT_FILE = "stopwords.json"

with open(FILE_DIR + STOPWORDS) as stopwordfile:
    stopwords = stopwordfile.read().splitlines()

[stopwords.append(c) for c in string.punctuation]
[stopwords.append(c+c) for c in string.punctuation]

print(stopwords)

with open(FILE_DIR + DOCUMENT_VOCAB) as vocabfile:
    vocab = {k: int(v) for (v, k, x) in [line.split() for line in vocabfile]}

stopword_list = [0, 1, 2, 3]
for stopword in stopwords:
    if stopword in vocab:
        stopword_list.append(vocab[stopword])
    if any(x for x in stopword if x.isalpha()):
        if stopword[0].isalpha():
            stop_cap = stopword.capitalize()
            if stop_cap in vocab:
                stopword_list.append(vocab[stop_cap])
        if len(stopword) > 1:
            stop_up = stopword.upper()
            if stop_up in vocab:
                stopword_list.append(vocab[stop_up])

stopword_list.sort()
print(len(stopword_list))

with open(FILE_DIR + OUTPUT_FILE, "w") as output:
    json.dump(stopword_list, output)
