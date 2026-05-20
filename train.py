import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Note: Using standard imblearn library for SMOTE to balance and enrich minority class boundaries
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    import os
    print("Installing imbalanced-learn dependency to handle class augmentation...")
    os.system("pip install imbalanced-learn")
    from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def build_advanced_ml_suite():
    print("🚀 Initializing High-Accuracy Multi-Algorithm Optimization Engine...")
    
    # 1. Load Dataset & Select Feature Subsets matching our Flask UI
    raw_data = load_breast_cancer()
    df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
    y = raw_data.target  # 0 = Malignant, 1 = Benign
    
    selected_features = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness"
    ]
    X = df[selected_features].copy()
    
    # 2. Train-Test Split (80/20) with Stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Handle Minority Class Augmentation using SMOTE on the Training set
    # This prevents any synthetic data leakage into our true Test validation pool
    print("📈 Applying SMOTE data augmentation to balance sample footprints...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # 4. Define Hyperparameter Grids for our 4 target ML Estimators
    # These configurations sweep through depths, weights, and regularizations to squeeze out maximum accuracy
    algorithms_config = {
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [4, 6, 8, 10],
                "classifier__criterion": ["gini", "entropy"]
            }
        },
        "Gradient_Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "classifier__n_estimators": [50, 100, 150],
                "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "classifier__max_depth": [3, 4, 5]
            }
        },
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=5000, random_state=42),
            "params": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "classifier__solver": ["saga", "lbfgs"]
            }
        },
        "Support_Vector_Machine": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                "classifier__C": [0.1, 1.0, 10.0, 100.0],
                "classifier__gamma": ["scale", "auto", 0.01, 0.1]
            }
        }
    }
    
    best_overall_score = -1.0
    best_overall_pipeline = None
    best_algorithm_name = ""
    performance_summary = []
    
    # 5. Iterative Grid Search Evaluation loop across all 4 architectures
    for name, config in algorithms_config.items():
        print(f"\n⚡ Optimizing Grid Space parameters for: {name}...")
        
        # Build independent pipeline utilizing standard scaling to protect model convergence
        base_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', config["model"])
        ])
        
        # 5-Fold Stratified Cross-Validation
        grid_search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=config["params"],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_res, y_train_res)
        optimized_pipeline = grid_search.best_estimator_
        
        # Run predictions on completely unseen validation data
        test_predictions = optimized_pipeline.predict(X_test)
        
        # Extract Performance Metrics
        acc = accuracy_score(y_test, test_predictions)
        prec = precision_score(y_test, test_predictions)
        rec = recall_score(y_test, test_predictions)
        f1 = f1_score(y_test, test_predictions)
        
        performance_summary.append({
            "Algorithm": name,
            "Accuracy": f"{acc * 100:.2f}%",
            "Precision": f"{prec * 100:.2f}%",
            "Recall": f"{rec * 100:.2f}%",
            "F1-Score": f"{f1 * 100:.2f}%"
        })
        
        # Log performance to terminal
        print(f"   ↳ Optimal Params: {grid_search.best_params_}")
        print(f"   ↳ Out-of-Sample Accuracy: {acc * 100:.2f}% | F1-Score: {f1 * 100:.2f}%")
        
        # Keep track of the champion model across all evaluations
        if acc > best_overall_score:
            best_overall_score = acc
            best_overall_pipeline = optimized_pipeline
            best_algorithm_name = name

    # 6. Print Comprehensive Comparative Summary Table
    summary_df = pd.DataFrame(performance_summary)
    print("\n" + "="*23 + " BENCHMARK RADAR REPORT " + "="*23)
    print(summary_df.to_string(index=False))
    print("="*70)
    
    # 7. Serialize and Save the Champion Pipeline Artifact
    model_filename = 'cancer_model.pkl'
    print(f"\n🏆 Champion Model Selected: {best_algorithm_name} ({best_overall_score * 100:.2f}% Accuracy)")
    print(f"Saving compiled pipeline payload asset to: {model_filename}")
    
    joblib.dump(best_overall_pipeline, model_filename)
    
    # Verify performance integrity with absolute target names classification matrix
    final_preds = best_overall_pipeline.predict(X_test)
    print("\n🔍 Final Model Classification Matrix Context:")
    print(classification_report(y_test, final_preds, target_names=raw_data.target_names))

if __name__ == "__main__":
    build_advanced_ml_suite()