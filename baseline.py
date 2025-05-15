import pandas as pd

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit

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

    subject_labels = y.reset_index().groupby('subject')['group'].first()

    # Stratified split at the subject level
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_subjects_idx, test_subjects_idx = next(splitter.split(subject_labels.index, subject_labels.values))

    train_subjects = subject_labels.index[train_subjects_idx]
    test_subjects = subject_labels.index[test_subjects_idx]

    # Mask by subject for train/test
    train_mask = X.index.get_level_values('subject').isin(train_subjects)
    test_mask = X.index.get_level_values('subject').isin(test_subjects)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    majority_class = y_train.value_counts().idxmax()
    y_majority_pred = [majority_class] * len(y_test)
    majority_accuracy = accuracy_score(y_test, y_majority_pred)

    print(f"Training on {len(X_train)} samples from a total of {len(X_train) + len(X_test)}")

    # TODO: sweep over parameters
    clf = RandomForestClassifier(random_state=23)
    clf.fit(X_train, y_train)

    # === Evaluation ===
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # binarize the labels
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test).ravel()

    majority_class_bin = lb.transform([majority_class])[0][0]
    y_majority_proba = [majority_class_bin] * len(y_test)

    majority_auc = roc_auc_score(y_test_bin, y_majority_proba)
    auc = roc_auc_score(y_test_bin, y_proba)

    print("\n=== Majority Voting ===")
    print(f"Majority class: {majority_class}")
    print(f"Majority Accuracy: {majority_accuracy:.4f}")
    print(f"Majority AUC: {majority_auc:.4f}")

    print("\n=== Random Forest ===")
    print(classification_report(y_test, y_pred))
    print(f"AUC: {auc}")

if __name__ == "__main__":
    # This file assumes you have a calculated feature extraction information.
    main()
