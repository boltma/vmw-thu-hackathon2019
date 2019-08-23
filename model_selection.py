# !/usr/bin/python
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import warnings
from models import texts
warnings.filterwarnings('ignore', category=UserWarning, module='gensim')

from gensim import corpora, models


def lda_model_selection(corpus, id2word, r):
    print('Selecting LDA models...')
    model_list = []
    coherence_values = []
    for num_topics in r:
        print('Number of topics: %d' % num_topics)
        model = models.LdaModel(corpus, num_topics=num_topics, id2word=id2word,
                                alpha='auto', eta='auto', minimum_probability=0.001, passes=10)
        model_list.append(model)
        coherence_model = models.CoherenceModel(model=model, texts=Text, dictionary=dictionary, coherence='c_v')
        v = coherence_model.get_coherence()
        coherence_values.append(v)
        print('Coherence value: %f' % v)

    plt.plot(r, coherence_values)
    plt.xlabel('Num Topics')
    plt.ylabel('Coherence score')
    plt.legend('coherence_values', loc='best')
    plt.show()


if __name__ == '__main__':
    Text = texts.Text
    dictionary = corpora.Dictionary(Text)
    corpus = [dictionary.doc2bow(text) for text in Text]
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    lda_model_selection(corpus_tfidf, dictionary, range(3, 30, 3))

    print('Getting HDP model coherence...')
    hdp = models.HdpModel(corpus_tfidf, id2word=dictionary)
    coherence_model = models.CoherenceModel(model=hdp, texts=Text, dictionary=dictionary, coherence='c_v')
    v = coherence_model.get_coherence()
    print('Coherence value: %f' % v)
