import pandas as pd
import numpy as np
import os
from src.DimondPricePrediction.logger import logging
from sklearn.model_selection import train_test_split
from src.DimondPricePrediction.exception import customexception
from pathlib import Path
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.DimondPricePrediction.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

import sys
class DataTransformation:
    def _init__(self):
        self.data_transformation_config=DataTransformation()
    

        
    def get_data_transformation(self,train_path):
        try:
            logging.info("data transformation initiated")
            train_data=pd.read_csv(train_path)

            categorical_columns=train_data.select_dtypes(include='object').columns
            numerical_columns=train_data.select_dtypes(exclude='object').columns
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            categorical_pipeline=Pipeline(  
                                            steps=[
                                                        ('imputer',SimpleImputer(strategy='most_frequent')),
                                                        ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories]))
                                                    ]
    
                                            )
            preprocessor=ColumnTransformer(steps=[('numerical_pipeline',numerical_pipeline,numerical_columns),
                                                ('categorical_pipeline',categorical_pipeline,categorical_columns)]

                                            )     
            return preprocessor    
        except Exception as e:
            raise customexception(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            logging.info("Train and test reading done")
            logging.info(train_data.head(2))
            logging.info("test_data.head(2)")
            preprocessing_obj=self.get_data_transformation(train_path)
            target_column="price"
            drop_column="price"
            x_train=train_data.drop(labels=target_column,axis=1)
            x_test=test_data.drop(labels=target_column,axis=1)
            y_train=train_data.target_column
            y_test=test_data.target_column
            x_train_preprocessed=preprocessing_obj.fit_transform(x_train)
            x_test_preprocessed=preprocessing_obj.transform(x_test)
            logging.info("train test preprocessing done")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
        except Exception as e:
            raise customexception(e,sys)


        
