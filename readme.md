#  DocNet: Semantic Structure in Inductive Bias Detection Models
This repository holds the raw partisan topic data and cleaned code used in the EMNLP 2024 submission "DocNet: Semantic Structure in Inductive Bias Detection Models"

For the BASIL dataset see: https://github.com/launchnlp/BASIL

## Contents
* \*.pkl: news articles for each topic
* inductive_pipeline.py  and inductive_pipeline_basil.py: pipeline for experimenting across embedding configurations 
* Document-level Bias by LLM.ipynb: notebook for LLM bias detection
* newsnet_utils.py: functions for cleaning up the data
* analysis_utils.py: functions for varying embedding configurations
* run_gcngae.py: functions for creatting autoencoder models (with GCN encoder)

## Usage:
1. Clean the data using `process_data.py`
2. Update scripts/notebooks with processed data file paths
2. Train desired model or full experimental configuration using the respective `run_{insert modelname}` functions or run `inductive_pipeline.py` as main