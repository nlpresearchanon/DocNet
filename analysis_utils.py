import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def arithmeticMean(results):
    #take avg by domain
    results = pd.DataFrame(results.groupby('domain')['embeddings'].apply(lambda x: np.mean(list(x), axis=0)))
    results['doc_id'] = results.index
    return results

def topicNormDiff_domainAvg(results):
    #get avg by topic and avg by domain
    topicavg = pd.DataFrame(results.groupby('topic')['embeddings'].apply(lambda x: np.mean(list(x), axis=0)))
    results['embeddings'] = results.apply(lambda x: x.embeddings-topicavg['embeddings'][x.topic], axis = 1)
    results = pd.DataFrame(results.groupby('domain')['embeddings'].apply(lambda x: np.mean(list(x), axis=0)))
    results['doc_id'] = results.index
    return results

def l2normDiff_domainAvg(results):
    #hypothetically find extremes by highlighting "difference from avg" via frobenius norm
    topicavg = pd.DataFrame(results.groupby('topic')['embeddings'].apply(lambda x: np.mean(list(x), axis=0)))
    results['embeddings'] = results.apply(lambda x: x.embeddings-topicavg['embeddings'][x.topic], axis = 1)
    results = pd.DataFrame(results.groupby('domain')['embeddings'].apply(lambda x: np.linalg.norm(np.array(list(x)), axis = 0)))
    results['doc_id'] = results.index
    return results

def no_aggregation(results):
    return results


# embedding filter functions
def no_filter(results, big_df):
    return results

def filterMultArticles(results, big_df):
    #filter to only embeddings from domains with at least 3 articles
    domain_groups = big_df.groupby(['domain']).count() 
    sub_domain = domain_groups[domain_groups['base_url'] >= 3] 
    results = results[results['domain'].isin(list(sub_domain.index))]
    return results

def filterMultTopic(results, big_df):
    #filter to only domains with at least 1 article per topic
    domain_groups = big_df.groupby(['domain','topic']).count().groupby(level=['domain']).count()
    sub_domain = domain_groups[domain_groups['base_url'] >= 3] 
    results = results[results['domain'].isin(list(sub_domain.index))]
    return results

def filterAfg(results, big_df):
    #filter to only afg 
    results = results[results['topic']=='afg']
    return results


def filterOath(results, big_df):
    #filter to only oath
    results = results[results['topic']=='oath']
    return results

def filterVaccine(results, big_df):
    #filter to only vax 
    results = results[results['topic']=='vaccine']
    return results
