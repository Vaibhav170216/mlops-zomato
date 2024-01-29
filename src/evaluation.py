import logging 
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


class MSE(Evaluation):

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating MSE.")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calcualting MSE {e}.")
            raise e

class R2(Evaluation):

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating R2 score.")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2: {r2}.")
            return r2
        except Exception as e:
            logging.error(f"Error in calcualting R2 score {e}.")
            raise e

class RMSE(Evaluation):

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calcualting RMSE score.")
            rmse = mean_squared_error(y_true, y_pred, squared = False)
            logging.info(f"RMSE: {rmse}.")
            return rmse
        except Exception as e:
            logging.error(f"Error in calcualting RMSE {e}.")
            raise e


