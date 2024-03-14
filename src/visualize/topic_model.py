"""
Try to understand the distribution of X by topic modeling
"""

import os
from typing import List

from stefutil import *
from src.util.ner_example import *
from src.data_util import NerDatasetStats

if __name__ == '__main__':

    dnm = 'conll2003-no-misc'
    ts = 'test'
    ds = NerDatasetStats.from_dir_name(dataset_name=dnm, dir_name=ts)
    egs: List[NerReadableExample] = ds.egs
    # sic(len(egs), egs[:10])
    sents = [eg.sentence for eg in egs]
    sic(sents[:10])

    lst_toks = TextPreprocessor()(texts=sents)
    sic(len(lst_toks), lst_toks[:10])

    def try_topic_model():
        from gensim.corpora.dictionary import Dictionary
        from gensim.models import LdaMulticore
        dictionary = Dictionary(lst_toks)
        dictionary.filter_extremes(no_below=5, no_above=0.5)  # drop tokens in less than 3 sentences & in more than 50% of sentences
        # sic(dictionary.token2id)
        sic(len(dictionary))

        corpus = [dictionary.doc2bow(doc) for doc in lst_toks]
        sic(len(corpus), corpus[:10])

        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=50, num_topics=5, passes=10, workers=os.cpu_count() - 1)
        sic(lda_model.print_topics(num_topics=-1))
    try_topic_model()
