import os
import sys
import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Saves a Python object using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a Python object using pickle.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple models with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: Dictionary of model names and model objects
        param: Dictionary of model names and their hyperparameter grids
    
    Returns:
        Dictionary with model names as keys and test R² scores as values
    """
    try:
        report = {}

        # Get model names and model objects
        model_names = list(models.keys())
        model_objects = list(models.values())

        for i in range(len(model_objects)):
            model = model_objects[i]
            model_name = model_names[i]  
            para = param[model_name]

            logging.info(f"Training {model_name}...")
            print(f"   Training {model_name}...")

            # Hyperparameter tuning with GridSearchCV
            if para:  # Only do grid search if there are params
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
                logging.info(f"{model_name} best params: {gs.best_params_}")

            # Train the model with best parameters
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R² scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test score in report
            report[model_name] = test_model_score  # FIXED: Use model_name variable

            logging.info(f"{model_name} - Train R²: {train_model_score:.4f}, Test R²: {test_model_score:.4f}")
            print(f"{model_name}: Train R²={train_model_score:.4f}, Test R²={test_model_score:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)