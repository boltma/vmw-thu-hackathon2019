# -*- coding:utf-8 -*-
import json
import networkx as nx
import matplotlib.pyplot as plt


def load_data(patent, citation, related, fname):
    try:
        with open(fname) as file:
            data = json.load(file)
        for d in data:
            patent.append((d['id'], d['patent_code'], d['application_number']))
            citation.append(d['citations'])
            related.append(d['related'])
    except FileNotFoundError:
        print('Data file'+fname+'not found')


# load data from data set
patents = []
citations = []
related = []

load_data(patents, citations, related, r"data\patents deep learning.json")
load_data(patents, citations, related,
          r"data\patents interactive visualization.json")
load_data(patents, citations, related, r"data\patents knowledge graph.json")
load_data(patents, citations, related, r"data\patents machine learning.json")
load_data(patents, citations, related, r"data\patents recommender system.json")
load_data(patents, citations, related, r"data\patents search engine.json")

G = nx.Graph()
node_patent = []
node_citation = []
node_related = []

# create nodes and edges
for patent in patents:
    node_patent.append(patent[0])

for citation in citations:
    for cite in citation:
        if cite not in node_citation:
            node_citation.append(cite)

for rela in related:
    for r in rela:
        if r not in node_related:
            node_related.append(r)

node_patent_cited = []

for node in node_citation:
    for patent in patents:
        if node in patent:
            node_citation.remove(node)
            node_patent_cited.append(patent[0]+'_')
            G.add_edge(patent[0], patent[0]+'_')

for node in node_related:
    for patent in patents:
        if node in patent:
            node_related.remove(node)
            node_patent_cited.append(patent[0]+'_')
            G.add_edge(patent[0], patent[0]+'_')

G.add_nodes_from(node_patent)
G.add_nodes_from(node_citation)

for i in range(len(node_patent)):
    for j in range(len(citations[i])):
        G.add_edge(node_patent[i], citations[i][j])
    for k in range(len(related[i])):
        G.add_edge(node_patent[i], related[i][k])


# draw figure
pos = nx.spring_layout(G, k=0.03, fixed=node_patent) # k=0.02 when draw figure of one subject

plt.figure(figsize=(50, 50)) # (30,30) when draw figure of one subject

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=node_citation,
    node_color='b',
    node_size=50,
    alpha=0.3
)

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=node_related,
    node_color='g',
    node_size=50,
    alpha=0.3
)

nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.2)

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=node_patent+node_patent_cited,
    node_color='red',
    node_size=300,
    with_labels=True
)

nx.draw_networkx_labels(G, pos, labels=dict(
    zip(node_patent+node_patent_cited, node_patent+node_patent_cited)))

plt.savefig(r'visualization\citation graph\total.png')
