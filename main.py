import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import pprint
from decision_tree import ID3, predict


def evaluate_model(tree, X_test, y_test, attributes):
    predictions = []
    for _, row in X_test.iterrows():
        prediction = predict(tree, row, attributes)
        predictions.append(prediction)

    y_pred = pd.Series(predictions)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred
    }


def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath, header=None,
                       names=["top-left", "top-middle", "top-right",
                              "middle-left", "middle-middle", "middle-right",
                              "bottom-left", "bottom-middle", "bottom-right",
                              "class"])

    data['class'] = data['class'].map({'positive': 1, 'negative': 0})

    X = data.drop(columns=['class'])
    y = data['class']

    return X, y


def split_data(X, y, test_size=0.3, val_size=0.15, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state)

    test_val_ratio = test_size / (test_size + val_size)
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=1-test_val_ratio, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X, y = load_and_prepare_data("data/tic-tac-toe.data")
    attributes = list(X.columns)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.2, val_size=0.2, random_state=42)

    train_objects = X_train.copy()
    train_objects['class'] = y_train.values
    objects = train_objects.values.tolist()

    tree = ID3(attributes, objects, max_depth=5)

    # Evaluate on validation set
    print("\nEvaluating on validation set:")
    val_results = evaluate_model(tree, X_val, y_val, attributes)
    print(f"Validation Accuracy: {val_results['accuracy']:.2%}")
    print("\nConfusion Matrix:")
    print(val_results['confusion_matrix'])
    print("\nClassification Report:")
    print(val_results['classification_report'])

    # Evaluate on test set
    print("\nEvaluating on test set:")
    test_results = evaluate_model(tree, X_test, y_test, attributes)
    print(f"Test Accuracy: {test_results['accuracy']:.2%}")
    print("\nConfusion Matrix:")
    print(test_results['confusion_matrix'])
    print("\nClassification Report:")
    print(test_results['classification_report'])

    # Optionally print the tree structure
    if input("\nPrint tree structure? (y/n): ").lower() == 'y':
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(tree)
