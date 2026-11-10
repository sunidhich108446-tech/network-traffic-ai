from pathlib import Path
import joblib 
import pandas as pd 
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score 
from src.features import prepare_features 

PROJECT_ROOT = Path(__file__).resolve().parent.parent 
DATA_PATH = PROJECT_ROOT/"data"/"raw"/"final.csv"
MODEL_PATH = PROJECT_ROOT/ "models"/"latency_model.joblib"

def main():
    df= pd.read_csv(DATA_PATH)
    X,y = prepare_features(df)
    split_index= int(len(X) * 0.8)

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    # Load saved model 
    saved_model= joblib.load(MODEL_PATH)
    model = saved_model["model"]
    features= saved_model["features"]
    X_test= X_test[features]
    predictions =  model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse =  mean_squared_error(y_test,predictions) **0.5
    r2_Score = r2_score(y_test,predictions)

    print("Model Evaluation Results:")
    print("-------------------------")
    print(f"MAE: {mae:.3f} ms")
    print(f"RMSE:{rmse:.3f} ms")
    print(f"R2 : {r2_Score:.3f} ms")

    if __name__== "__main__":
        main()
