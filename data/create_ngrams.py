"""Generate Ngrams"""

from __future__ import absolute_import, division, print_function

import gc
import glob
import json
import pickle
from collections import Counter
from multiprocessing import Pool

STOPWORD_FILE = "stopwords.json"
NGRAMS = 3
FEATURE = 'document_sequence'
LABEL = 'answer_ids'
PATH_PREFIX = ""
TRAIN_FILE = "train"
VALIDATIION_FILE = "validation"
TEST_FILE = "test"
TARGET = TEST_FILE

with open(PATH_PREFIX + '/' + STOPWORD_FILE) as stopword_file:
    stopwords = json.load(stopword_file)


class NgramExtractor(object):
    def __init__(self, feature=FEATURE, ngram_range=NGRAMS):
        self.feature = feature
        self.ngram_range = ngram_range

    def __call__(self, filename):
        print(f"Getting ngram set from {filename}")
        with open(filename, 'r') as file:
            sequences = [json.loads(line)[self.feature] for line in file]

        gc.collect()

        fileset = Counter()
        for sequence in sequences:
            for i in range(2, self.ngram_range + 1):
                ngrams = zip(*[sequence[j:] for j in range(i)])
                for ngram in ngrams:
                    if all(x not in stopwords for x in ngram):
                        fileset[ngram] += 1

        del sequences
        gc.collect()

        print(f"Extracted {len(fileset)} ngrams from {filename}")
        most_common = fileset.most_common(250000)
        del fileset
        gc.collect()
        return most_common


def create_ngram_set(filenames, feature=FEATURE, ngrams=NGRAMS):
    """
    Extract a set of n-grams from a list of files containing feature where
    feature is a list of ints.

    create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    # Create set of unique n-grams from the training set.
    with Pool() as pool:
        new_sets = pool.map(NgramExtractor(feature, ngrams), filenames)

    gc.collect()

    ngram_freq = Counter()
    for new_set in new_sets:
        for ngram in new_set:
            ngram_freq[ngram[0]] += ngram[1]

    del new_sets
    gc.collect()

    sorted_tokens = [x for (x, v) in ngram_freq.most_common(500000)]

    del ngram_freq
    gc.collect()

    return sorted_tokens


def main():
    filenames = [f"{PATH_PREFIX}/{file}" for file in glob.glob1(
        PATH_PREFIX, f"{TARGET}*.json")]

    sorted_ngrams = create_ngram_set(filenames, feature=FEATURE, ngrams=NGRAMS)

    with open(f"{PATH_PREFIX}/{NGRAMS}-grams-{TARGET},pkl", 'wb') as savefile:
        pickle.dump(sorted_ngrams, savefile)


if __name__ == "__main__":
    main()
