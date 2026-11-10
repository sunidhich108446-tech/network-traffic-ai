from pathlib import Path
import sys
import joblib
import pandas as pd 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.features import prepare_features 

#File Paths 
DATA_PATH = PROJECT_ROOT/"data"/"raw"/"final.csv"
MODEL_DIR =PROJECT_ROOT/"models"
MODEL_PATH=  MODEL_DIR/"latency_model.joblib"

def main():
    df= pd.read_csv(DATA_PATH)
    print("Dataset loaded :",df.shape)

    # 2. Features aur target prepare karo 
    X,y = prepare_features(df)
    split_index = int(len(X)*0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Model Pipeline 
    model=  Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor",XGBRegressor(n_estimators=300,learning_rate=0.03,max_depth=3,subsample=0.8,colsample_bytree=0.8,reg_lambda=1,objective="reg:squarederror",random_state=42,n_jobs=-1)
)

    ])

    #Training of Model 
    model.fit(X_train,y_train)
    print("Model Training complete ")

    # Test data Predictions
    predictions= model.predict(X_test)
    mae= mean_absolute_error(y_test,predictions)
    rmse = mean_squared_error(y_test,predictions) **0.5
    r2= r2_score(y_test,predictions)

    print("TEST RESULTS:")
    print(f"MAE: {mae: .3f} ms")
    print(f"RMSE:{rmse: .3f} ms")
    print(f"R2: {r2: .3f} ms")

    #Savin trained model
    MODEL_DIR.mkdir(parents =True, exist_ok= True)
    joblib.dump({
    "model": model,
    "features": list(X.columns),
    "target": "target_rtt_30s"
}, MODEL_PATH)

    print(f"Model saved at: {MODEL_PATH}")


if __name__=="__main__":
        main()
