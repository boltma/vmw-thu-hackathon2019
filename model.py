# !/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import gensim
import re
from gensim import corpora, models, similarities
from pprint import pprint
import warnings
import json
import spacy

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import networkx as nx

import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def remove_stopwords(doc):
    print('Removing stopwords...')
    stop_list = gensim.parsing.preprocessing.STOPWORDS
    stop_list = stop_list | frozenset(
        ['example', 'embodiment', 'embodiments', 'data', 'user', 'include',
         'includes'])
    return [[word for word in line.strip().lower().split() if (
        word not in stop_list and len(word) > 3)] for line in doc]


def make_trigrams(texts):
    print('Making trigrams...')
    # Build the bigram and trigram models. Higher threshold fewer phrases.
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=80)
    trigram = gensim.models.Phrases(bigram[texts], threshold=80)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


nlp = spacy.load('en', disable=['parser', 'ner'])


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    print('Lemmatizing...')
    lemmatized_texts = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        lemmatized_texts.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return lemmatized_texts


def train_lda_model(corpus, num_topics, id2word):
    print('Training LDA model...')

    lda_output = open('models/lda.txt', 'w')

    print('LDA Model:', file=lda_output)
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=id2word,
                          passes=2, alpha='auto', eta='auto', eval_every=1,
                          minimum_probability=0.001)

    doc_topic = [doc_t for doc_t in lda[corpus]]
    print('\nDocument-Topic:', file=lda_output)
    pprint(doc_topic, stream=lda_output)
    for doc_topic in lda.get_document_topics(corpus):
        print(doc_topic, file=lda_output)

    for topic_id in range(num_topics):
        print('\nTopic', topic_id, file=lda_output)
        pprint(lda.show_topic(topic_id), stream=lda_output)

    vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(vis, 'visualization/LDA_Visualization.html')

    similarity = list(similarities.MatrixSimilarity(lda[corpus]))
    print('\nSimilarity:', file=lda_output)
    pprint(similarity, stream=lda_output)
    draw_graph(adj_matrix=similarity, threshold=0.6,
               file_name='visualization/lda_similarity.png')


def train_hdp_model(corpus, num_topics, id2word):
    print('Training HDP model...')

    hdp_output = open('models/hdp.txt', 'w')

    hdp = models.HdpModel(corpus_tfidf, id2word=dictionary)
    topic_result = [a for a in hdp[corpus_tfidf]]

    print('HDP Model:', file=hdp_output)
    pprint(topic_result, stream=hdp_output)

    print('\nHDP Topics:', file=hdp_output)
    print(hdp.print_topics(num_topics=num_topics, num_words=5), file=hdp_output)

    similarity = list(similarities.MatrixSimilarity(hdp[corpus_tfidf]))
    print('\nSimilarity:', file=hdp_output)
    pprint(similarity, stream=hdp_output)
    draw_graph(similarity, 0.6, 'visualization/hdp_similarity.png')


def draw_graph(adj_matrix, threshold, file_name):
    plt.figure(dpi=128, figsize=(32, 32))
    G = nx.Graph()
    node_num = len(patent_id)
    for i in range(node_num):
        G.add_node(patent_id[i])
        for j in range(i):
            if adj_matrix[i][j] > threshold:
                G.add_edge(patent_id[i], patent_id[j], weight=adj_matrix[i][j])

    pos = nx.circular_layout(G)
   # pos = nx.kamada_kawai_layout(G)
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_cmap=plt.cm.Blues, width=2, alpha=0.5,
                           edge_color=weights)

    plt.savefig(file_name)
    plt.close()


if __name__ == '__main__':

    data = []
    with open('data/patents_all.json') as file:
        data = json.load(file)

    patent_id = []
    doc = []
    # 提取专利描述，切分成单词
    for patent in data:
        abstract = patent['description']
        abstract = re.sub('[^a-zA-Z]', ' ', abstract)
        doc.append(abstract)
        patent_id.append(patent['id'])

    texts = remove_stopwords(doc)
    texts = make_trigrams(texts)
    texts = lemmatization(texts)

    text_output = open('models/texts.txt', 'w')
    print('Text = ', file=text_output)
    pprint(texts, stream=text_output)

    num_topics = 20

    dictionary = corpora.Dictionary(texts)
    V = len(dictionary)
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpus_tfidf = models.TfidfModel(corpus)[corpus]

   # print('TF-IDF:')
   # for c in corpus_tfidf:
   #     print(c)

    """
    sys.stdout = open('interactive_visualization_lsi.txt', 'w')
    print('\nLSI Model:')
    lsi = models.LsiModel(
        corpus_tfidf, num_topics=num_topics, id2word=dictionary)
    topic_result = [a for a in lsi[corpus_tfidf]]
    pprint(topic_result)
    print('LSI Topics:')
    pprint(lsi.print_topics(num_topics=num_topics, num_words=5))
    similarity = similarities.MatrixSimilarity(
        lsi[corpus_tfidf])   # similarities.Similarity()
    print('Similarity:')
    similarity_matrix = list(similarity)
    pprint(similarity_matrix)
    """
    train_lda_model(corpus_tfidf, num_topics, dictionary)
    train_hdp_model(corpus_tfidf, num_topics, dictionary)
