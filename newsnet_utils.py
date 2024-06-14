import pandas as pd, numpy as np, spacy, re, itertools
import contractions
from collections import Counter
import math
import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.preprocessing import MinMaxScaler

import torch
from torch_geometric.data import Data

import gensim.models.doc2vec
assert gensim.models.doc2vec.FAST_VERSION > -1
from gensim.models.doc2vec import TaggedDocument

import networkx as nx

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from umap.umap_ import UMAP
from hdbscan import flat

np.random.seed(28)

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("sentencizer")
nlp.add_pipe("merge_entities")
nlp.max_length = 3000000

def process_text(text): 
    #the basic text processing
    #spacy settings
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace(
        "&lt;", "<").replace("Loading Something is loading", "") #filler from business inside articles
    text = re.sub(r"http\S+", "", text)
    #change new lines into sentences
    text = re.sub("\n", ". ", text)
    text = re.sub("\.\.", ".", text)
    text = re.sub("\. \.", ".", text)
    text = contractions.fix(text)
    text = nlp(text) #run through senticizer and merge entities pipe
    sentences = [i for i in text.sents if len(i) >2 ]
    #remove  filler sentences (avoid learning the article metadata)
    junk_list = ["this advertisement","the unsubscribe link", "your junk folder", "your inbox", "free streaming app",
                "download and start watching", "click here", "subscribe", "load error", "replay video"] 
    sentences = [i for i in sentences if not any(junk in str(i).lower() for junk in junk_list)]
    return sentences

class AugmentedTextGraph: 
    #make sure you call process_text(text) first to get the sentences used as input
    def __init__(self):
        self.word_to_id = {}
        self.unique_tokens = []

    def create_augmented_text_graph(self, sentences):
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = [[res[1] for res in sid.polarity_scores(sentence.text).items()] for sentence in sentences]
        lemmas, vectors = self._process_sentences(sentences)
        self.unique_tokens = list(np.unique([item for sublist in lemmas for item in sublist]))
        self.word_to_id = dict(zip(self.unique_tokens, range(len(self.unique_tokens))))
        self.node_attr = self._create_node_attr(lemmas, vectors, sentiment_scores)
        self.edge_index, self.edge_attr = self._create_text_graph(lemmas)
        data = Data(x=torch.tensor(self.node_attr, dtype=torch.float),
                    edge_index = torch.tensor(self.edge_index, dtype=torch.long),
                    edge_attr = torch.tensor(self.edge_attr, dtype=torch.float))
        return data

    def _process_sentences(self, sentences):
        lemmas = [[token.lemma_.strip().lower() for token in sent if token.lemma_ if 
                ( not (token.is_stop) and not (token.is_punct) and not (token.is_space) and not (token.like_num))
                ] for sent in sentences]
        screen_mask = [i for i in range(len(lemmas)) if len(lemmas[i])>1]
        lemmas = [lemmas[i] for i in screen_mask]
        vectors = [[token.vector for token in sent if token.lemma_ if 
                ( not (token.is_stop) and not (token.is_punct) and not (token.is_space) and not (token.like_num))
                ] for sent in sentences]
        vectors = [vectors[i] for i in screen_mask]
        return lemmas, vectors
    
    def _create_text_graph(self, text):
        network = Counter()
        unigram_freq = Counter()
        for sent in text:
            window_ids = [self.word_to_id[word] for word in sent]
            network.update(itertools.product(window_ids,window_ids))
            unigram_freq.update(window_ids)
        edge_x =[]
        edge_y = []
        edge_attr = []
        for item in network.items():
            if item[0][0] != item [0][1]:
                edge_weight = self._pmi(item[0][0], item[0][1], unigram_freq, network)
                # add first directed edge
                edge_x.append(item[0][0])
                edge_y.append(item[0][1])
                edge_attr.append(edge_weight)
                # add second directed edge to make it undirectional
                edge_x.append(item[0][1])
                edge_y.append(item[0][0])
                edge_attr.append(edge_weight)
                
        # only include positive PMI scores 
        # (not true PPMI though since otherwise we would just be changing all neg values to 0)
        screen_mask = [i for i in range(len(edge_attr)) if edge_attr[i] > 0]
        if len(screen_mask) == 0: #no co-occurring words
            raise ValueError ("There are no co-occurring words in this document")
        edge_x = [edge_x[i] for i in screen_mask]
        edge_y = [edge_y[i] for i in screen_mask]
        edge_attr = [edge_attr[i] for i in screen_mask]
                
        return np.array([edge_x, edge_y]),  edge_attr
    
    def _pmi(self, word1, word2, unigram_freq, bigram_freq):
        prob_word1 = unigram_freq[word1] / float(sum(unigram_freq.values()))
        prob_word2 = unigram_freq[word2] / float(sum(unigram_freq.values()))
        prob_word1_word2 = bigram_freq[(word1, word2)] / float(sum(bigram_freq.values()))
        return np.log2(prob_word1_word2/float(prob_word1*prob_word2)) / -np.log2(prob_word1_word2)
    
    def _create_node_attr(self, lemmas, vectors, sentiment_scores):
        vectors = [[np.append(vec, sentiment_scores[i]) for vec in vectors[i]] for i in range(len(vectors))]
        all_lemmas = [item for sublist in lemmas for item in sublist]
        all_vectors = np.array([item for sublist in vectors for item in sublist])
        x = []
        for lemma in self.unique_tokens:
            indices = [i for i in range(len(all_lemmas)) if all_lemmas[i] == lemma]
            x.append(np.mean(all_vectors[indices, :], axis=0))
        x = np.array(x)
        return x

    def makenx_graph(self, sentences):
        #outputs an undirected Networkx graph for input into graph2vec (reindexes nodes)
        _ = self.create_augmented_text_graph(sentences)
        edge_index = self.edge_index
        edge_attr = self.edge_attr #ppmi
        node_attr = self.node_attr #spacy lm vectors with sentiment
        token_map = {v: k for k, v in self.word_to_id.items()} 
        #check if node indices and num of nodes actually align (necessary for graph2vec)
        if len(np.unique(edge_index))!= (edge_index.max()+1): 
            #need to reindex bc there are fewer nodes than they are indexed
            #only include nodes that have an edge
            reduce_token_map = {k:v for k, v in token_map.items() if k in np.unique(edge_index)} 
            #original key: new key
            new_index = {k:i for k,i in zip(reduce_token_map.keys(), range(0, len(np.unique(edge_index))))} 
            token_map = {v:token_map[k] for k, v in new_index.items()} #new token map with reindexed keys
            #relabel edges
            edge_index[0] = np.array([new_index[i] for i in edge_index[0]])
            edge_index[1] = np.array([new_index[i] for i in edge_index[1]])
            #only keep relevant nodes per original node index
            node_attr = node_attr[list(reduce_token_map.keys())]

        #make graph
        G =nx.Graph()
        G.add_nodes_from([(int(i), {"feature":token_map[i], "node_attr": node_attr[i]}) for i in np.unique(edge_index)]) 
        G.add_edges_from([(int(i[0]), int(i[1]),
            {"edge_attr":i[2]}) for i in zip(edge_index[0], edge_index[1], edge_attr)])
        return G
    
class AugmentedTextGraphSentiment(AugmentedTextGraph):
    def __init__(self):
        super().__init__()
        
    def _create_node_attr(self, lemmas, vectors, sentiment_scores):
        vectors = [[sentiment_scores[i] for vec in vectors[i]] for i in range(len(vectors))]
        all_lemmas = [item for sublist in lemmas for item in sublist]
        all_vectors = np.array([item for sublist in vectors for item in sublist])
        x = []
        for lemma in self.unique_tokens:
            indices = [i for i in range(len(all_lemmas)) if all_lemmas[i] == lemma]
            # mean and standard deviation of sentiment
            x.append(np.concatenate((np.mean(all_vectors[indices, :], axis=0), np.std(all_vectors[indices, :], axis=0))))
        x = np.array(x)
        return x

##data setup for doc2vec
class NewsCorpus():
        def __init__(self, data, train = True) :
            self.data = data #list of sentences
            self.train = train #tag document or not
        def preprocess(self, sentences): 
            #adapted from augmentedtextgraph (gensim also has basic simple preprocess)
            lemmas = [[token.lemma_.strip().lower() for token in sent if token.lemma_ if 
                    ( not (token.is_stop) and not (token.is_punct) and not (token.is_space) and not (token.like_num))
                    ] for sent in sentences]
            lemmas = [x for sublist in lemmas for x in sublist]
            return lemmas 
        def __iter__(self):
            for i in range(len(self.data)):
                    if self.train: 
                            yield TaggedDocument(self.preprocess(self.data[i]), [i])
                    else: #new data so don't tag
                            yield self.preprocess(self.data[i])