import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class MLOpsWorkshop:
    def __init__(self):
        self.mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment("Classification_Workshop")
        self.models = {}
        self.results = {}
        
    def generate_dataset(self):
        """Generate synthetic classification dataset"""
        print("Generating synthetic dataset...")
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/dataset.csv', index=False)
        
        return X, y
    
    def preprocess_data(self, X, y):
        """Split and scale the data"""
        print("Preprocessing data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model with MLflow tracking"""
        print("Training Random Forest...")
        
        with mlflow.start_run(run_name="Random_Forest"):
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            mlflow.log_param("model_type", "Random Forest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            mlflow.sklearn.log_model(model, "model")
            
            self.models['Random Forest'] = model
            self.results['Random Forest'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        return model
    
    def train_svm(self, X_train, y_train, X_test, y_test):
        """Train SVM model with MLflow tracking"""
        print("Training SVM...")
        
        with mlflow.start_run(run_name="SVM"):
            model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            )
            
            mlflow.log_param("model_type", "SVM")
            mlflow.log_param("kernel", "rbf")
            mlflow.log_param("C", 1.0)
            mlflow.log_param("gamma", "scale")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            mlflow.sklearn.log_model(model, "model")
            
            self.models['SVM'] = model
            self.results['SVM'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        return model
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression model with MLflow tracking"""
        print("Training Logistic Regression...")
        
        with mlflow.start_run(run_name="Logistic_Regression"):
            model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
            
            mlflow.log_param("model_type", "Logistic Regression")
            mlflow.log_param("max_iter", 1000)
            mlflow.log_param("solver", "lbfgs")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            mlflow.sklearn.log_model(model, "model")
            
            self.models['Logistic Regression'] = model
            self.results['Logistic Regression'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        return model
    
    def compare_models(self):
        """Compare all models and select the best one"""
        print("\nModel Comparison Results:")
        print("=" * 60)
        
        results_df = pd.DataFrame(self.results).T
        print(results_df.round(4))
        
        best_model_name = results_df['f1_score'].idxmax()
        best_f1_score = results_df['f1_score'].max()
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best F1 Score: {best_f1_score:.4f}")
        
        self.create_comparison_plot(results_df)
        
        return best_model_name, self.models[best_model_name]
    
    def create_comparison_plot(self, results_df):
        """Create visualization comparing model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            results_df[metric].plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral', 'lightgreen'])
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_workshop(self):
        """Run the complete MLOps workshop"""
        print("Starting MLOps Classification Workshop")
        print("=" * 50)
        
        X, y = self.generate_dataset()
        X_train, X_test, y_train, y_test, scaler = self.preprocess_data(X, y)
        
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_svm(X_train, y_train, X_test, y_test)
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        
        best_model_name, best_model = self.compare_models()
        
        print(f"\nWorkshop completed!")
        print(f"Access MLflow UI at: {self.mlflow_uri}")
        print(f"Best model ({best_model_name}) saved and tracked in MLflow")

if __name__ == "__main__":
    workshop = MLOpsWorkshop()
    workshop.run_workshop()