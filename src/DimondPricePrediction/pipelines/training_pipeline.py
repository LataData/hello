from src.DimondPricePrediction.components.data_ingestion import DataIngestion
import pandas as pd
import numpy as np
import os
from src.DimondPricePrediction.logger import logging
from sklearn.model_selection import train_test_split
from src.DimondPricePrediction.exception import customexception
from pathlib import Path
from dataclasses import dataclass
import sys

obj=DataIngestion()
obj.initiate_data_ingestion()
