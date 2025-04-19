import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv("data/tic-tac-toe.data", header=None, names=["top-left", "top-middle", "top-right", "middle-left", "middle-middle", "middle-right", "bottom-left", "bottom-middle", "bottom-right", "class"])


    data['class'] = data['class'].map({'positive': 1, 'negative': 0})
    X = data.drop(columns=['class'])
    y = data['class']

    # Podział na zbiory treningowy, walidacyjny i testowy
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)