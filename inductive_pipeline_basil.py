# pipeline to train embedding and run supervised model (logit) without seeing test data

from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sentence_transformers import SentenceTransformer

import pandas as pd
import numpy as np
import torch
import pickle
from collections import Counter

from analysis_utils import arithmeticMean, topicNormDiff_domainAvg, l2normDiff_domainAvg, no_aggregation
from analysis_utils import no_filter, filterAfg, filterOath, filterVaccine, filterMultTopic, filterMultArticles

import math
import itertools
import warnings
warnings.filterwarnings('ignore')

from random import shuffle

from inductive_pipeline import run_doc2vec, run_graph2vec, run_graphdoc2vec, run_sbert, run_gae, make_label_name

pd.options.mode.chained_assignment = None


def iterate_sup_models(df, embeddingOptions, labelType, modelType, 
                       filterOptions, aggregationOptions, save_path, kwargs):
    ''' Iterate through configurations and save results of predictions
    Filters data, generates an unsupervised embedding, and trains a supervised model
    Modified from inductive_pipeline's version for BASIL data
    Parameters:
        df: processed data
        embeddingOptions: the embedding model
        labelType: name of label for file saving (e.g. article)
        modelType: name of supervised model for file saving
        filterOptions: methods of filtering the data
        aggregationOptions: methods of aggregating the embedding
        save_path: where to save results
        kwargs: embedding model arguments
    '''
    seed=28
    full_scores = {}
    full_results = {}
    df2 = df.copy()
    biasOrdinal1 = {"FAR LEFT":1, "LEFT":2, "LEFT-CENTER":3, "CENTER":4,
                                "RIGHT-CENTER":5, "RIGHT":6, "FAR RIGHT":7}
    #merge labels to extreme
    biasOrdinal3 =  {"LEFT":2, "LEFT-CENTER":1, "CENTER":1,
            "RIGHT-CENTER":1, "RIGHT":2}
    # clean out left-center into left
    biasOrdinal2 = {"LEFT":1, "LEFT-CENTER":1, "CENTER":2,
        "RIGHT-CENTER":3, "RIGHT":3}
        
    for filter_function, aggregation_method, biasTuple in itertools.product(
                                                            filterOptions, aggregationOptions, 
                                                            zip([biasOrdinal1, biasOrdinal2, biasOrdinal3],
                                                                ["no_merge", "aggregate", "extreme"])):
        # filter data, make train test split, embed
        biasOrdinal, labelName = biasTuple
        df = filter_function(df2, df2)
        # make train test split
        if aggregation_method.__name__ == "no_aggregation":
            #make train test split
            df_train, df_test = model_selection.train_test_split(
                df, train_size=0.80, test_size=0.20, random_state=seed)
        else:
           return "error, shouldn't be trying to aggregate basil data"
        columns_of_interest = ['doc_id', 'domain', 'topic', 'Bias', 'text']
        full_results[filter_function.__name__]={'test_data': df_test[columns_of_interest], 
                                                'train_data': df_train[columns_of_interest]}
        for embedding_function, common_kwargs in zip(embeddingOptions, kwargs):
            # create embedding
            all_embeddings = embedding_function(df_train, df_test, **common_kwargs)
            # train embeddings and run supervised model for each sub-model type
            for model_name, results0 in all_embeddings.items():
                biasOrdinal, labelName = biasTuple
                key_name = (model_name, filter_function.__name__, aggregation_method.__name__, labelName) 
                #convert train and test data to aggregation and format needed
                X_list = []
                y_list = []
                for dataset in ["train", "test"]: # process data for train, then test data
                    results = results0[dataset][results0[dataset]['embeddings'].notnull()]
                    # bring labels back in to embeddings
                    results = results.merge(df[['doc_id', 'domain', 'topic']], on="doc_id", how="inner")
                    #convert to array for easier math
                    if type(results['embeddings'].iloc[0]) is torch.Tensor:       
                            results['embeddings'] = results['embeddings'].apply(lambda x: x.numpy())     
                    #run aggregation on embeddings (no agg is an option, which means everything will be at article level)
                    results = aggregation_method(results)
                    results = results.merge(df[['doc_id', 'domain','Bias', 'topic']],on='doc_id', how = "inner") 

                    #replace with ordinal labels
                    results['BiasOrd']= results['Bias'].replace(biasOrdinal)
                    embeddings = np.stack(results['embeddings'].array)
                    
                    X_list.append(embeddings)
                    y_list.append(results['BiasOrd'])
                # run logit model
                pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, multi_class="multinomial"))
                pipe.fit(X_list[0], y_list[0])
                # get all predictions
                y_pred = pipe.predict(X_list[1])
                y_predprob = pipe.predict_proba(X_list[1])
                accuracy = accuracy_score(y_list[1], y_pred)
                f1 = f1_score(y_list[1], y_pred, average ="weighted")
                f1macro = f1_score(y_list[1], y_pred, average ="macro")
                full_scores[key_name] = {'accuracy':accuracy, 'f1weight': f1, 'f1macro':f1macro}                   
                
                full_labels = list(set([*y_list[1], *y_pred]))
                full_names = [ make_label_name(i) for i in itertools.product(full_labels, full_labels)]
                cm = confusion_matrix(y_list[1], y_pred, labels = full_labels).ravel() 
                cmvalues = {full_names[i]: cm[i] for i in range(len(cm))}
                count_true = Counter(y_list[1])
                count_true = {str(k)+"_true": count_true[k] for k in count_true.keys()}
                count_pred = Counter(y_pred)
                count_pred = {str(k)+"_pred": count_pred[k] for k in count_pred.keys()}
                count_train = Counter(y_list[0])
                count_train = {str(k)+"_train": count_train[k] for k in count_train.keys()}
                full_results[key_name] = {'pred_y':y_pred, 'true_y':y_list[1],'prob_y': y_predprob,
                                                     'test_doc_id':results['doc_id'], #mcnemar test data
                                                     'train_y': y_list[0], 
                                                     'confusion_matrix': cmvalues,
                                                     'count_true': count_true,
                                                     'count_pred': count_pred,
                                                     'count_train': count_train} 

    #save options used into dict for array generation
    filterKey = {filterOptions[v].__name__:v for v in range(len(filterOptions))}
    aggregationKey = {aggregationOptions[v].__name__: v for v in range(len(aggregationOptions))}

    scoreKey = {'accuracy':0, 'f1weight': 1, 'f1macro':2}
    labelKey = {"no_merge":0, "aggregate":1, "extreme":2}
    #save results to file
    with open("{}_scores_{}_{}.pickle".format(save_path, modelType, labelType), "wb") as f:
        pickle.dump ((filterKey, aggregationKey, labelKey, scoreKey, full_scores), f)
    with open("{}_predictions_{}_{}.pickle".format(save_path, modelType, labelType), "wb") as f:
        pickle.dump ((filterKey, aggregationKey, labelKey, scoreKey, full_results), f)
    print("trained and scores saved")
    return True

def cleanBasilLabels(df, labelType):
    if labelType == "article":
        df['Bias'] = df['article_bias']
    elif labelType == 'domain':
        df['Bias'] = df['domain_bias']
    return df

if __name__=="__main__":
    d2vargs = dict( # doc2vec
            window = 10,
            alpha = .05, 
            seed = 28,
            epochs = 50,
            negative = 5,
            hs = 0,
            vector_size = 128,
            min_count = 3, 
            workers = 5)
    g2vargs = dict( # graph2vec
            workers = 5,
            dimensions = 128, 
            epochs = 50,  
            wl_iterations = 2)
    d2gvargs = dict(
            doc2vec = d2vargs,
            graph2vec = g2vargs)
    sbtargs = dict(meaningless_arg = None)
    aeargs = dict( # autoencoders
            norm = True,
            early_stopping =False,
            num_features = 128,
            common_kwargs = dict(
            max_epochs = 50, #50
            min_epochs = 5,
            accelerator= "gpu",
            devices = 1))
    
    save_path = "results_inductive_2024/supervised_basil_" #path for ouput pickles
    modelType = "logit"
    common_kwargs = [
        d2vargs, 
        g2vargs, 
        d2gvargs,
        sbtargs, 
        aeargs
        ]
    embeddingOptions = [
        run_doc2vec, 
        run_graph2vec, 
        run_graphdoc2vec, 
        run_sbert, 
        run_gae
        ]
    # filtering and aggregating embeddings are not valid for BASIL 
    filterOptions =  [no_filter ] 
    aggregationOptions = [no_aggregation]
    #load BASIL data
    basil = "data/processed/basil_data.pkl"
    df = pd.read_pickle(basil)
    df['_id'] = df['url']    #urls are definitely unique
    df['doc_id'] = df['url']
    df['domain'] = df['source']
    df['topic'] = df['triplet-uuid']

    labelTypes =  ["article", "domain"]
   
    for labelType in labelTypes:
        sub_df = cleanBasilLabels(df, labelType)
        # run through all models
        iterate_sup_models(sub_df, embeddingOptions, labelType, modelType, filterOptions, 
                           aggregationOptions, save_path, common_kwargs)