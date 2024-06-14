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
from run_gcngae import AEDataModule, VariationalGCNEncoderNorm, GCNEncoderNorm, VariationalGCNEncoder, GCNEncoder, LitAutoencoder
from torch_geometric.nn import  VGAE, GAE, global_mean_pool
from torch_geometric.loader import DataLoader as geo_DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

import math
import itertools
import warnings
warnings.filterwarnings('ignore')

#doc2vec packages
import gensim.models.doc2vec
assert gensim.models.doc2vec.FAST_VERSION > -1
from gensim.models.doc2vec import Doc2Vec
from random import shuffle

#karateclub packages
from karateclub import Graph2Vec

pd.options.mode.chained_assignment = None

# no test bleed doc2vec
def run_doc2vec(train_text_df, test_text_df, **common_kwargs):    
    full_data = list(train_text_df["doc2vec_data"])
    all_embeddings = {}
    models = [Doc2Vec(dm = 1, **common_kwargs), #DM
        Doc2Vec(dm = 0, **common_kwargs)] #DBOW
        
    shuffle(full_data)
    for item in zip(models, ["doc2vec_dm", "doc2vec_dbow"]):
        model = item[0]
        model.build_vocab(full_data)
        all_embeddings[item[1]] = {}
        print("vocab built")
        print("Training "+ item[1])
        model.train(full_data, epochs = model.epochs, total_examples = model.corpus_count)
        #get train embeddings
        print("getting embeddings")
        embedding = [model.dv[data[1][0]] for data in train_text_df['doc2vec_data']] #get embedding from the tag
        results=pd.DataFrame()
        results[['doc_id','domain_label']] = train_text_df[['_id','Bias']]
        results['embeddings'] = embedding
        all_embeddings[item[1]]["train"] = results
        #get test embeddings
        embedding = [model.infer_vector(text[0]) for text in test_text_df["doc2vec_data"]]
        results=pd.DataFrame()
        results[['doc_id','domain_label']] = test_text_df[['_id','Bias']]
        results['embeddings'] = embedding
        all_embeddings[item[1]]["test"] = results
        # print(results.head())
    return all_embeddings


# no test bleed graph2vec
def run_graph2vec(train_text_df, test_text_df,**common_kwargs): 
    # use node word in g2v_nodewords model (no point in running with sentiment data since g2v doesnt't use node attributes)
    all_embeddings = {}
    #load data from previously processed
    #drop any documents that were skipped(no co-occurring words)
    list_df = [train_text_df, test_text_df]
    list_graphs = [0,0]
    for i in range(2):
        print("dropping {} graphs".format(len(list_df[i][list_df[i]["graph2vec_data"].isna()])))
        list_df[i] = list_df[i][list_df[i]["graph2vec_data"].notnull()]
        list_graphs[i] = list_df[i]["graph2vec_data"]
        print(list_df[i].shape)
    for item in zip(['feature', None, 'node_attr'], ['g2v_nodewords', 'g2v_no_features', 'g2v_spacy']):
        #run graph2vec to build embedding (this does not factor in node or edge features) 
        print(item[1])
        modelnodes = Graph2Vec(attributed = item[0], **common_kwargs)  #can mess with size of embedding with no minimal difference in accuracy downstream (it's labeling everything mostly as right)
        modelnodes.fit(list_graphs[0]) #fit on train data
        #infer embeddings
        print("getting embeddings")
        all_embeddings[item[1]] = {}
        for i,dataset in zip(range(2),["train","test"]):
            g2v_embedding = modelnodes.infer(list_df[i]["graph2vec_data"])
            results=pd.DataFrame()
            results[['doc_id','domain_label']] = list_df[i][['_id','Bias']]
            results['embeddings'] = list(g2v_embedding)
            all_embeddings[item[1]][dataset] = results
    return all_embeddings

def run_graphdoc2vec(train_text_df, test_text_df,**common_kwargs):
    # merges graph and doc2vec embeddings
    doc_embeddings = run_doc2vec(train_text_df, test_text_df, **common_kwargs["doc2vec"])
    graph_embeddings = run_graph2vec(train_text_df, test_text_df, **common_kwargs["graph2vec"])
    # make all model variations
    all_embeddings = {}
    for doc_name, graph_name in itertools.product(doc_embeddings.keys(), graph_embeddings.keys()):
        combine_key = f"{doc_name}_{graph_name}"
        all_embeddings[combine_key] = {}
        combine_train = doc_embeddings[doc_name]["train"].merge(
            graph_embeddings[graph_name]["train"][["doc_id", "embeddings"]],on="doc_id", how = "inner") 
        combine_train["embeddings"] = combine_train.apply(lambda x: np.concatenate((x.embeddings_x, x.embeddings_y)), axis = 1)
        combine_test = doc_embeddings[doc_name]["test"].merge(
            graph_embeddings[graph_name]["test"][["doc_id", "embeddings"]],on="doc_id", how = "inner") 
        combine_test["embeddings"] = combine_test.apply(lambda x: np.concatenate((x.embeddings_x, x.embeddings_y)), axis = 1)
        all_embeddings[combine_key]["train"] = combine_train
        all_embeddings[combine_key]["test"] = combine_test
    return all_embeddings

def run_sbert(train_text_df, test_text_df, **common_kwargs):
    key = "sbert"
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    
    print("getting embeddings")
    all_embeddings = {'sbert': {}}
    for key, df in zip(["train", "test"], [train_text_df, test_text_df]):
        sbert_embedding = []
        for sentences in df["transformer_data"]:
            embeddings = model.encode(sentences) #outputs a (#sentences, 768) shape array
            avg_embedding = np.mean(embeddings, 0)#average out into a vector of size 768
            sbert_embedding.append(avg_embedding)
        results=pd.DataFrame()
        results[['doc_id','domain_label']] = df[['_id','Bias']]
        results['embeddings'] = sbert_embedding
        all_embeddings['sbert'][key] = results
    return all_embeddings

def run_gae(train_text_df, test_text_df, **common_kwargs):
    # run autoencoders inductively
    early_stopping = common_kwargs['early_stopping'] 
    num_features = common_kwargs['num_features']
    norm = common_kwargs['norm']
    common_kwargs = common_kwargs["common_kwargs"]
    # remove null data
    screened_text_df = train_text_df.copy()
    og_data = len(screened_text_df)
    screened_text_df = screened_text_df[screened_text_df["autoencode_data"].notnull()]
    test_text_df = test_text_df[test_text_df["autoencode_data"].notnull()]
    print("dropped {} null documents".format(og_data-len(screened_text_df)))
    all_embeddings = {}
    #run gcn encoder graph autoencoder and variational autoencoder with inner product decoder
    for col_name in ["autoencode_data", "autoencode_data_sent"]:
        dataset = list(screened_text_df[col_name])
        test_dataset = list(test_text_df[col_name])
        dm = AEDataModule(dataset)
        for variational in [True, False]:
            if variational: #run vGAE
                key = "VGAE"+col_name
                if norm:
                    encoder = VariationalGCNEncoderNorm(num_features)
                else:
                    encoder = VariationalGCNEncoder(num_features)
                model = LitAutoencoder(VGAE, encoder)
                log_path = "lightning_logs/vgae"+col_name
            else: 
                key = "GAE"+col_name
                if norm: 
                    encoder = GCNEncoderNorm(num_features)
                else:
                    encoder = GCNEncoder(num_features)
                model = LitAutoencoder(GAE,encoder)
                log_path = "lightning_logs/gae"+col_name
            print(key)
            lr_monitor = LearningRateMonitor(logging_interval='epoch') #so that learning rate is logged in tensorboard
            callbacks = [lr_monitor]
            if early_stopping:
                callbacks.append(EarlyStopping(monitor="val_loss", mode="min"))
            tb_logger = pl_loggers.TensorBoardLogger(save_dir = log_path)

            trainer = Trainer(logger = tb_logger, 
                        callbacks= callbacks,  
                        enable_checkpointing = False, 
                        profiler = False, 
                        **common_kwargs) 

            trainer.fit(model, dm) 
            #get embeddings
            print("getting embeddings")
            all_embeddings[key] = {}
            for k, ds, df in zip(['train', 'test'], [dataset, test_dataset], [screened_text_df, test_text_df]):
                test = geo_DataLoader(ds, 1, shuffle=False)
                ae_embeddings = []
                with torch.no_grad():
                    for data in test:
                        z = model.model.encode(data.x, data.edge_index, data.edge_attr, data.batch)
                        z = global_mean_pool(z, data.batch, 2)
                        ae_embeddings.append(z[0])
                results=pd.DataFrame()
                results[['doc_id','domain_label']] = df[['_id','Bias']]
                results['embeddings'] = list(ae_embeddings)
                all_embeddings[key][k] = results   
    return all_embeddings  
 
def make_label_name(labeltuple):
    return(f'true{labeltuple[0]}pred{labeltuple[1]}')

def iterate_sup_models(df, embeddingOptions, labelType, modelType, filterOptions, aggregationOptions, save_path, kwargs):
    ''' Iterate through configurations and save results of predictions
    Filters data, generates an unsupervised embedding, and trains a supervised model
    Parameters:
        df: processed data
        embeddingOptions: the embedding model
        labelType: name of label for file saving 
        modelType: name of supervised model for file saving
        filterOptions: methods of filtering the data
        aggregationOptions: methods of aggregating the embedding
        save_path: where to save results
        kwargs: embedding model arguments
    '''
    seed=28
    
    df['doc_id'] = df['base_url']
    df['_id'] = df['base_url'] # only truly unique value since some _ids are duplicates

    #create label merging patterns 
    biasOrdinal1 = {"FAR LEFT":1, "LEFT":2, "LEFT-CENTER":3, "CENTER":4,
                                    "RIGHT-CENTER":5, "RIGHT":6, "FAR RIGHT":7}
    #merge labels to l,c,r
    biasOrdinal2 = {"FAR LEFT":1, "LEFT":1, "LEFT-CENTER":1, "CENTER":2,
                "RIGHT-CENTER":3, "RIGHT":3, "FAR RIGHT":3}
    #merge labels to extreme
    biasOrdinal3 =  {"FAR LEFT":2, "LEFT":2, "LEFT-CENTER":1, "CENTER":1,
                "RIGHT-CENTER":1, "RIGHT":2, "FAR RIGHT":2}
    full_scores = {}
    full_results = {}
    df2 = df.copy()
    settings = itertools.product(filterOptions, aggregationOptions, 
                                                                zip([biasOrdinal1, biasOrdinal2, biasOrdinal3], 
                                                                ["no_merge", "aggregate", "extreme"]))
    settings_list = [(i, y, z) for i, y, z in settings]
    restart_num = 59 # where to restart
    for (filter_function, aggregation_method, biasTuple) in settings_list[restart_num:]:
         # filter data , make train test split, embed
        biasOrdinal, labelName = biasTuple
        df = filter_function(df2, df2)
        # make train test split
        if aggregation_method.__name__ == "no_aggregation":
            #make train test split
            df_train, df_test = model_selection.train_test_split(
                df, train_size=0.80, test_size=0.20, random_state=seed)
        else:
            # if aggregation is happening we need to split by domains and then merge original article back in
            # flaw is since domains share articles, there may still be test data bleed
            domain_df = df.drop_duplicates(subset = ['domain'])["domain"]
            print("num_domains ", len(domain_df))
            df_train, df_test = model_selection.train_test_split(
                domain_df, train_size=0.80, test_size=0.20, random_state=seed)
            # merge all articles that correspond to the training domain back in
            df_train = df.merge(df_train, on = "domain", how = "right")
            df_test = df.merge(df_test, on = "domain", how = "right")
        columns_of_interest = ['doc_id', 'domain', 'topic', 'Bias', 'text']
        data_key = ('data', filter_function.__name__, aggregation_method.__name__) 
        full_results[data_key]={'test_data': df_test[columns_of_interest], 
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
                    #merge domain bias label back in (gets rid of duplicate entries of domain)
                    domain_bias = pd.DataFrame(df[['domain','Bias']]).drop_duplicates()
                    results = results.merge(domain_bias,on="domain", how = "left") 

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
                with open("{}_append_scores_{}_{}.pickle".format(save_path, modelType, labelType), "ab+") as f:
                    pickle.dump ({key_name: full_scores[key_name]}, f)
                with open("{}_append_predictions_{}_{}.pickle".format(save_path, modelType, labelType), "ab+") as f:
                    pickle.dump ({key_name: full_results[key_name]}, f)

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
            devices = [0]))
    
    save_path = "results_inductive/" #path for output pickles

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
    filterOptions =  [
        no_filter, filterAfg, filterOath,
          filterVaccine, filterMultTopic, filterMultArticles
          ]
    aggregationOptions = [no_aggregation, arithmeticMean, topicNormDiff_domainAvg, l2normDiff_domainAvg]
    #load processed data
    afg = "data/processed/afghanistan_withdrawal_website_text_29DEC22.pkl"
    oath = "data/processed/oath_keepers_website_text_29DEC22.pkl"
    vaccine = "data/processed/military_vaccine_merged_3NOV22.pkl"
    
    files = [ 
            afg, oath,
              vaccine]
    topic = [ 
        "afg", "oath", 
        "vaccine"]
    big_df = pd.DataFrame()
    for i in range(len(files)):
        df = pd.read_pickle(files[i])
        df['topic'] = str(topic[i])
        big_df = pd.concat([big_df, df])

    big_df = big_df.drop_duplicates(subset = 'base_url') #drop duplicated articles
    print(big_df.shape)

    #rename _id as the url since some of the topics don't have a mongo id
    big_df['clean_url'] = big_df['base_url'].apply(lambda x: x.replace('https://', ""))
    big_df['clean_url'] = big_df['clean_url'].apply(lambda x: x.replace('http://', ""))
    big_df = big_df.drop_duplicates(subset = 'clean_url') #drop duplicated articles jic
    print(big_df.shape)

    big_df['_id'] = big_df['base_url']

    labelType = "oglabel"
    iterate_sup_models(big_df, embeddingOptions, labelType, modelType, filterOptions, aggregationOptions, save_path, common_kwargs)