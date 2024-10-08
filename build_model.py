import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from common import DATA_KEY_COLUMN, DATA_FILEPATH, DATA_FILE_ENCODING, MAPPING_FILEPATH, MAPPING_VALUE_COLUMN, \
    MAPPING_KEY_COLUMN, MAPPING_FILE_ENCODING, get_logging
from data_cleaning import process_data


def build_and_train_model(df: pd.DataFrame, target_columns: list) -> [MLPClassifier, dict]:
    # Check for missing values
    if df.isnull().values.any():
        print("Warning: Data contains missing values. Consider handling them before training.")

    # Separate features and targets
    X = df.drop(columns=target_columns)
    y = df[target_columns]

    # Split data into 80% training and 20% test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the MLPClassifier with the hidden layers
    clf = MLPClassifier(hidden_layer_sizes=(100, 32, 8, 3), random_state=42, max_iter=1000)

    # Train the model for each target column and store scores
    scores = {}
    for column in target_columns:
        clf.fit(X_train_scaled, y_train[column])
        y_pred = clf.predict(X_test_scaled)
        scores[column] = accuracy_score(y_test[column], y_pred)

    return clf, scores


if __name__ == '__main__':
    logger = get_logging()
    try:
        # Process the data using the cleaning function from the `data_cleaning.py`
        df_processed = process_data(DATA_FILEPATH, DATA_FILE_ENCODING, DATA_KEY_COLUMN, MAPPING_FILEPATH,
                                    MAPPING_FILE_ENCODING, MAPPING_KEY_COLUMN, MAPPING_VALUE_COLUMN)

        # Identify target columns containing the substring 'tnufa_endreasonName'
        target_columns = [col for col in df_processed.columns if DATA_KEY_COLUMN in col]

        # Build and train the model
        model, test_scores = build_and_train_model(df_processed, target_columns)

        for target, score in test_scores.items():
            logger.info(f"Model accuracy on the test set for {target}: {score:.4f}")
    except Exception as e:
        logger.info(f"An error occurred: {e}")
