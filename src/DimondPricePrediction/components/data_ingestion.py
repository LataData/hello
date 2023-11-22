import pandas as pd
import numpy as np
import os
from src.DimondPricePrediction.logger import logging
from sklearn.model_selection import train_test_split
from src.DimondPricePrediction.exception import customexception
from pathlib import Path
from dataclasses import dataclass
import sys
class DataIngestionConfig:
    raw_data_path:str=(os.path.join("artifacts","raw_data.csv"))
    train_data_path:str=os.path.join("artifacts","train_data.csv")
    test_data_path:str=os.path.join("artifacts","test_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
       

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            df=pd.read_csv(Path(os.path.join("notebooks/data","gemstone.csv")))
            logging.info("Read Data")
            df=df.drop(labels=["id"],axis=1)
            train_data,test_data=train_test_split(df,test_size=.25)
            logging.info("Train Test split done")
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("raw data artifact done")
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.train_data_path)),exist_ok=True)
            train_data.to_csv((self.ingestion_config.train_data_path),index=False)
            logging.info("train data artifact done")
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.test_data_path)),exist_ok=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("Data ingestion done")
            return(self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)

        except Exception as e:
            logging.info("exception during data ingestion")
            print(e)
            raise customexception(e,sys)