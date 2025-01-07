import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dataclasses import dataclass  # Importing dataclass
from src.exception import CustomException
from src.logger import logging
import pickle

@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info("Training the RandomForest model")
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            logging.info("Making predictions")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Model accuracy: {accuracy}")

            # Save the model
            os.makedirs(os.path.dirname(self.model_trainer_config.model_file_path), exist_ok=True)
            with open(self.model_trainer_config.model_file_path, "wb") as file:
                pickle.dump(model, file)

            logging.info("Model saved successfully")
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
