import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df, target_col):

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # split types
    cat = X.select_dtypes(include="str")
    num = X.select_dtypes(include=["int64", "float64"]).astype(np.float32)

    # encode
    X_enc = pd.get_dummies(cat)

    # concat
    X = np.concatenate([num.values, X_enc.values], axis=1)

    # normalize
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)

    return X.astype(np.float32), y.values.astype(np.float32)

def split(X, y, test_size=0.2, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    return (
        X_train,
        X_test,
        y_train.reshape(-1,1),
        y_test.reshape(-1,1)
    )