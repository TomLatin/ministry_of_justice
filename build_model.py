import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from data_cleaning import process_data, DATA_FILEPATH, DATA_FILE_ENCODING, MAPPING_FILEPATH, MAPPING_FILE_ENCODING, \
    DATA_KEY_COLUMN, MAPPING_KEY_COLUMN, MAPPING_VALUE_COLUMN


def build_and_train_model(df: pd.DataFrame, target_column: str):
    """
    Build, train, and evaluate an MLPClassifier model.

    Parameters:
    - df: The processed DataFrame with features and target column.
    - target_column: The column name which contains the target labels.

    Returns:
    - model: The trained MLPClassifier model.
    - score: The accuracy score of the model on the test set.
    """
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split data into 80% training and 20% test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define the MLPClassifier with the hidden layers
    clf = MLPClassifier(hidden_layer_sizes=(100, 32, 8, 3), random_state=42, max_iter=1000)

    # Train the model
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the accuracy
    score = accuracy_score(y_test, y_pred)

    return clf, score


if __name__ == '__main__':
    # Process the data using the cleaning function from the `data_cleaning.py`
    df_processed = process_data(DATA_FILEPATH, DATA_FILE_ENCODING, DATA_KEY_COLUMN, MAPPING_FILEPATH,
                                MAPPING_FILE_ENCODING, MAPPING_KEY_COLUMN, MAPPING_VALUE_COLUMN)
    # Define the column that we want to predict
    target_column = 'tnufa_endreasonName'

    # Build and train the model
    model, test_score = build_and_train_model(df_processed, target_column)

    print(f"Model accuracy on the test set: {test_score:.4f}")
