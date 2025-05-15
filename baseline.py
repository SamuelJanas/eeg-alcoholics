import pandas as pd

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    alcoholic_features = pd.read_csv("data/alcoholic_features.csv")
    control_features = pd.read_csv("data/control_features.csv")
    full_data = pd.concat([alcoholic_features, control_features])

    # Aggregate the data to include one patient trial as a single sample
    # we will ignore the sensor information
    numeric_cols = full_data.select_dtypes(include='number').columns.difference(['subject', 'trial'])
    X = full_data.groupby(['subject', 'trial'])[numeric_cols].agg(['mean', 'std', 'min', 'max'])

    X.columns = ['_'.join(col).strip() for col in X.columns.values]
    y = full_data.groupby(['subject', 'trial'])['group'].first()

    # TODO: parametrize this (or not)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, train_size=0.7)

    print(f"Training on {len(X_train)} samples from a total of {len(X_train) + len(X_test)}")

    # TODO: sweep over parameters
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # === Evaluation ===
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # This file assumes you have a calculated feature extraction information.
    main()
