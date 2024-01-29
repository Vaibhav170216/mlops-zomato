from zenml import pipeline
from steps.ingesting_data import ingest_data
from steps.cleaning_data import clean_data
from steps.model_training import model_train
from steps.evaluate import evaluate_model

@pipeline(enable_cache = False)
def train_pipeline(data_path: str):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = model_train(X_train, y_train)
    r2, rmse = evaluate_model(model, X_test, y_test)
