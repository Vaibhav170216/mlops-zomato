import logging
from typing import Union
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop(["restaurant name", "restaurant type", "online_order", "table booking", "cuisines type", "area", "local address"], axis = 1)
            data = data.rename(columns = {"rate (out of 5)":"rate",
                                         "avg cost (two people)":"cost",}) 
            data.select_dtypes(include = [np.number])                 
            data.dropna(inplace = True)
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing the data {e}.")
            raise e
        
class DataDivideStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(["cost"], axis = 1)
            y = data["cost"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in dividing data {e}.")
            raise e

class DataCleaning:

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data {e}.")
            raise e