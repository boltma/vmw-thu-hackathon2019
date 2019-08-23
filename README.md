# vmw-thu-hackathon2019
code for vmw-thu-hackathon2019 problem 5

original data stored in `/patent_doc`

Run `crawler.py` to get data from PatFT and store in `/data`, `model.py` for training model and `model_selection.py` for selecting LDA model. `model_selection.py` should be run after data has been collected and texts have been split by `model.py`. `citation_graph.py` generates citation graphs.

Models are stored in `/models` and visualization results could be found in `/visualization`

## Project Name

Analysis of patent structures in AI field

## Team

Fan Xiaoxiong, Li yuze, Ma Xiaoyang, You Yuyang

## Problem Statement

Analyze the given dataset of AI patents, and find citations and dependencies among the patents.

Citations could be found from official websites of USPTO, but dependencies shall be found by analyzing topics and finding similarities.

Visualize the found results.

## Data Sources

Original data are stored in `/patent_doc` in pdf forms, whose file names are patent numbers.

We get patent information by crawling from Patent Full-Text Database (PatFT), stored in `/data` as json forms.

## Design & Methodology

The whole project could be separated into four parts: data collection, data preprocessing, model training and selection, visualization.

- Data Collection: crawling necessary information from PatFT.
- Data Preprocessing: use regular expressions and split texts, remove stopwords and generate bigram/trigram models, and transform into TF-IDF models.
- Model Training and Selection: use gensim to train LDA & HDP models, and select by coherence score, we then use a joint model of both models
- Visualization: use pyLDAvis and networkx to generate graphs.

For more details, see `Report.docx`.

## Algorithm & Model

See report for details.

## Result

Visualized results are stored in `/visualization`

- Citation graphs: these graphs show citations and relationships (prior arts) between given patents.
- Similarity graphs: similarities are shown by colors of edges, higher similarities are illustrated by deeper colors.
- Topic visualization: a visualization of trained LDA model.

The models could achieve coherence scores of 0.55 and 0.71.