import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

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
    test_subjects = subject_labels.index[test_subjects_idx]    # Mask by subject for train/test
    train_mask = X.index.get_level_values('subject').isin(train_subjects)
    test_mask = X.index.get_level_values('subject').isin(test_subjects)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Encode string labels to numeric for XGBoost
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    majority_class = y_train.value_counts().idxmax()
    y_majority_pred = [majority_class] * len(y_test)
    majority_accuracy = accuracy_score(y_test, y_majority_pred)

    print(f"Training on {len(X_train)} samples from a total of {len(X_train) + len(X_test)}")

    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(random_state=23),
        'SVM': SVC(random_state=23, probability=True),
        'XGBoost': XGBClassifier(random_state=23, eval_metric='logloss')
    }

    # Train and evaluate each model
    results = {}
    for model_name, clf in models.items():
        print(f"\n=== Training {model_name} ===")
        
        # Use scaled features for SVM, original features for others
        if model_name == 'SVM':
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        elif model_name == 'XGBoost':
            # XGBoost needs numeric labels
            clf.fit(X_train, y_train_encoded)
            y_pred_encoded = clf.predict(X_test)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            y_proba = clf.predict_proba(X_test)[:, 1]
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]

        # Calculate metrics
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test).ravel()
        auc = roc_auc_score(y_test_bin, y_proba)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[model_name] = {
            'accuracy': accuracy,
            'auc': auc,
            'y_pred': y_pred,
            'y_proba': y_proba
        }

    # Display majority class baseline
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test).ravel()
    majority_class_bin = lb.transform([majority_class])[0][0]
    y_majority_proba = [majority_class_bin] * len(y_test)
    majority_auc = roc_auc_score(y_test_bin, y_majority_proba)

    print("\n=== Majority Voting Baseline ===")
    print(f"Majority class: {majority_class}")
    print(f"Majority Accuracy: {majority_accuracy:.4f}")
    print(f"Majority AUC: {majority_auc:.4f}")

    # Display results for all models
    for model_name, result in results.items():
        print(f"\n=== {model_name} Results ===")
        print(classification_report(y_test, result['y_pred']))
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"AUC: {result['auc']:.4f}")    # Summary comparison
    print("\n=== Model Comparison Summary ===")
    print(f"{'Model':<15} {'Accuracy':<10} {'AUC':<10}")
    print("-" * 35)
    print(f"{'Majority':<15} {majority_accuracy:<10.4f} {majority_auc:<10.4f}")
    for model_name, result in results.items():
        print(f"{model_name:<15} {result['accuracy']:<10.4f} {result['auc']:<10.4f}")
    
    # Create visualization
    plot_model_comparison(results, majority_accuracy, majority_auc)

def plot_model_comparison(results, majority_accuracy, majority_auc):
    """Create and save comparison plots for classical ML models - Accuracy and AUC only."""
    # Prepare data for plotting
    model_names = list(results.keys()) + ['Majority Baseline']
    accuracies = [results[name]['accuracy'] for name in results.keys()] + [majority_accuracy]
    aucs = [results[name]['auc'] for name in results.keys()] + [majority_auc]
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Classical Machine Learning Models Comparison', fontsize=16, fontweight='bold')
    
    # 1. Accuracy comparison bar plot
    bars1 = ax1.bar(model_names, accuracies, alpha=0.7, edgecolor='black', linewidth=1.2)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Color the baseline bar differently
    bars1[-1].set_color('lightcoral')
    bars1[-1].set_alpha(0.8)
    
    # Rotate x-axis labels for better readability
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. AUC comparison bar plot
    bars2 = ax2.bar(model_names, aucs, alpha=0.7, edgecolor='black', linewidth=1.2)
    ax2.set_title('Model AUC Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('AUC Score', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, auc in zip(bars2, aucs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Color the baseline bar differently
    bars2[-1].set_color('lightcoral')
    bars2[-1].set_alpha(0.8)
    
    # Rotate x-axis labels for better readability
    ax2.tick_params(axis='x', rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('figs/classical_models_accuracy_auc.png', dpi=300, bbox_inches='tight')
    plt.savefig('figs/classical_models_accuracy_auc.pdf', bbox_inches='tight')
    print(f"\nPlots saved to 'figs/classical_models_accuracy_auc.png' and 'figs/classical_models_accuracy_auc.pdf'")
    plt.show()

if __name__ == "__main__":
    main()
