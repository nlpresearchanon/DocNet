import pandas as pd, numpy as np, pickle

import torch
from typing import Optional, Tuple
from torch import Tensor

import torch.nn.functional as F
from torch.utils.data import random_split

from torch_geometric.loader import DataLoader
from torch_geometric.nn import  GraphConv, VGAE, GAE, GraphNorm, BatchNorm
from torch_geometric.utils import negative_sampling

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

from newsnet_utils import AugmentedTextGraph
# adapted from: https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial6/Tutorial6.ipynb

seed_everything(28, workers= True)

class AEDataModule(pl.LightningDataModule):
    #load pre-created pickle of augmented data graphs to dataloader

    def __init__(self, dataset = None, data_path = None, batch_size: int = 32, raw_path = None): 
        super().__init__()
        self.raw_path = raw_path
        self.data_path = data_path
        self.batch_size = batch_size
        self.dataset = dataset
    
    #save augmented text graph to a pickle file
    def make_dataset(self):
        # raw_path = "data/military_vaccine_website_text_3NOV22.pkl"
        # data_path = "data/augmented_text_graph_dataset_3NOV22.pkl"
        text_df = pd.read_pickle(self.raw_path)
        # text_df = text_df[text_df['text'].str.contains('(?=military)(?=.*vaccin)', regex=True)]
        screened_text_df = text_df[~text_df['Bias'].isna()]
        dataset = [AugmentedTextGraph().create_augmented_text_graph(text) for text in screened_text_df["text"]]

        print(len(dataset))
        with open(self.data_path, "wb") as f:
            pickle.dump(dataset, f)

    def prepare_data(self): #take raw pickeled dataframe of news and convert to list of augment data graph. Save as new pickle
        if self.raw_path is not None:
            self.make_dataset(self.raw_path, self.data_path)

    def setup(self, stage):
        if self.data_path is not None: #use a pickled dataset from make_dataset
            with open(self.data_path, "rb") as f:
                dataset = pickle.load(f)
        elif self.dataset is not None: #use premade dataset
            dataset = self.dataset

        if stage == "fit" or None: #don't split since it's unsupervised
            split = int(len(dataset)*.7)
            self.data_train, self.data_val = random_split(dataset, [split, len(dataset)-split])
        if stage =="test":
            self.data_test = dataset
        if stage =="predict":
            self.data_predict = dataset

    def train_dataloader(self):
        return DataLoader(self.data_train, self.batch_size, shuffle = True)
    def val_dataloader(self):
        return DataLoader(self.data_val, self.batch_size, shuffle = False)
    def test_dataloader(self):
        return DataLoader(self.data_test, self.batch_size, shuffle=False)
    def predict_dataloader(self):
        return DataLoader(self.data_predict, self.batch_size, shuffle=False)
    def teardown(self, stage:str):
        #cleanup return
        ...

class GCNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GraphConv(-1, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_weight=edge_weight) 
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        return x


class GCNEncoderNorm(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GraphConv(-1, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.norm1 = GraphNorm(hidden_channels)
        self.norm2 = GraphNorm(hidden_channels)
        self.norm3 = GraphNorm(hidden_channels)

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_weight=edge_weight) 
        x = self.norm1(x)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.norm2(x)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = self.norm3(x)
        return x

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GraphConv(-1, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3_mu = GraphConv(hidden_channels, hidden_channels)
        self.conv3_logstd = GraphConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_weight=edge_weight) 
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight) 
        x = x.relu()
        x_mu = self.conv3_mu(x, edge_index, edge_weight=edge_weight)
        x_logstd = self.conv3_logstd(x, edge_index, edge_weight=edge_weight)
        return x_mu, x_logstd

class VariationalGCNEncoderNorm(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GraphConv(-1, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3_mu = GraphConv(hidden_channels, hidden_channels)
        self.conv3_logstd = GraphConv(hidden_channels, hidden_channels)
        self.norm1 = GraphNorm(hidden_channels)
        self.norm2 = GraphNorm(hidden_channels)
        self.norm3_mu = GraphNorm(hidden_channels)
        self.norm3_logstd = GraphNorm(hidden_channels)
        
    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_weight=edge_weight) 
        x = self.norm1(x)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight) 
        # x = self.norm2(x)
        x = x.relu()
        x_mu = self.conv3_mu(x, edge_index, edge_weight=edge_weight)
        x_mu = self.norm3_mu(x)
        x_logstd = self.conv3_logstd(x, edge_index, edge_weight=edge_weight)
        x_logstd = self.norm3_logstd(x)
        return x_mu, x_logstd
#
#Generic lightning module
class LitAutoencoder(pl.LightningModule): 
    #adapted from torch_geometric.nn GAE
    def __init__(self, model, encoder, variational = False):
        super().__init__()
        self.model = model(encoder)
        self.automatic_optimization = True #performs backward/step/zero_grad for us)
        self.variational = variational #if true run variational GAE (VGAE)

    def forward(self, batch):
        z = self.model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch) 
        x_hat = self.model.decode(z, sigmoid =True)
        return x_hat

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor, num_nodes, 
                   neg_edge_index: Optional[Tensor] = None):
        """
        Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges
        """
        EPS = 1e-15
        pos_loss = -torch.log(
            self.model.decode(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.model.decode(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()
        loss = pos_loss + neg_loss
        if self.variational: #use variational auto encoder loss function
            loss = loss + (1 / num_nodes) * self.model.kl_loss()  # new line

        return loss, neg_edge_index
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)
        ##add in learning rate scheduler
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min'),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1} #check n times every interval
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        z = self.model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)  
        loss, neg_edge_index = self.recon_loss(z, batch.edge_index, batch.num_nodes)
        roc, ap = self.model.test(z, batch.edge_index, neg_edge_index)

        self.log_dict({"train_loss":loss, 
                    "train_roc": roc, 
                    "train_ap": ap}, 
            on_step = False, on_epoch = True, batch_size = 32)
        return loss
    
    def validation_step(self, batch, batch_idx):
        z = self.model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)  
        loss, neg_edge_index = self.recon_loss(z, batch.edge_index, batch.num_nodes)
        roc, ap = self.model.test(z, batch.edge_index, neg_edge_index)
        self.log_dict({"val_loss":loss, 
                    "val_roc": roc, 
                    "val_ap": ap}, 
            on_step = False, on_epoch = True, batch_size = 32)

    def test_step(self, batch, batch_idx):
        z = self.model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)  
        loss, neg_edge_index = self.recon_loss(z, batch.edge_index, batch.num_nodes)
        roc, ap = self.model.test(z, batch.edge_index, neg_edge_index)
        self.log_dict({"test_loss":loss, 
                    "test_roc": roc, 
                    "test_ap": ap}, 
            on_step = False, on_epoch = True, batch_size = 32)


if __name__ == "__main__":
    data_path = "processeddatapath.pkl"

    early_stopping =False
    num_features = 256
    max_epochs = 3
    min_epochs = 2
    variational = True

    text_df = pd.read_pickle(data_path)
    screened_text_df = text_df[~text_df['Bias'].isna()]
    dataset = screened_text_df['autoencode_data']

    dm = AEDataModule(dataset)

    if variational: #run vGAE
        encoder = VariationalGCNEncoder(num_features)
        model = LitAutoencoder(VGAE, encoder)
        log_path = "lightning_logs/vgae"
    else: 
        encoder = GCNEncoder(num_features)
        model = LitAutoencoder(GAE,encoder)
        log_path = "lightning_logs/gae"

    lr_monitor = LearningRateMonitor(logging_interval='epoch') 
    callbacks = [lr_monitor]
    if early_stopping:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min"))
    tb_logger = pl_loggers.TensorBoardLogger(save_dir = log_path)

    trainer = Trainer(logger = tb_logger, 
                max_epochs = max_epochs, 
                min_epochs =min_epochs, 
                callbacks= callbacks,
                accelerator= "cpu",
                devices = 1, 
                enable_checkpointing = False, 
                profiler = "simple") 

    trainer.fit(model, dm) 