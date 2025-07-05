import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class CreditScoringModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def generate_synthetic_data(self, n_samples=10000):
        np.random.seed(42)
        data = {
            'age': np.random.normal(40, 12, n_samples),
            'income': np.random.lognormal(10.5, 0.5, n_samples),
            'employment_length': np.random.exponential(5, n_samples),
            'debt_to_income_ratio': np.random.beta(2, 5, n_samples),
            'credit_history_length': np.random.gamma(2, 3, n_samples),
            'num_credit_accounts': np.random.poisson(3, n_samples),
            'num_late_payments': np.random.poisson(1, n_samples),
            'credit_utilization': np.random.beta(2, 3, n_samples),
            'num_inquiries': np.random.poisson(2, n_samples),
            'home_ownership': np.random.choice(['rent', 'own', 'mortgage'], n_samples, p=[0.3, 0.2, 0.5]),
            'loan_purpose': np.random.choice(['debt_consolidation', 'home_improvement', 'major_purchase', 'other'], 
                                           n_samples, p=[0.4, 0.2, 0.2, 0.2])
        }
        df = pd.DataFrame(data)
        df['age'] = np.clip(df['age'], 18, 80)
        df['income'] = np.clip(df['income'], 20000, 300000)
        df['employment_length'] = np.clip(df['employment_length'], 0, 40)
        df['debt_to_income_ratio'] = np.clip(df['debt_to_income_ratio'], 0, 1)
        df['credit_history_length'] = np.clip(df['credit_history_length'], 0, 30)
        df['num_credit_accounts'] = np.clip(df['num_credit_accounts'], 0, 15)
        df['num_late_payments'] = np.clip(df['num_late_payments'], 0, 10)
        df['credit_utilization'] = np.clip(df['credit_utilization'], 0, 1)
        df['num_inquiries'] = np.clip(df['num_inquiries'], 0, 10)
        credit_score = (
            0.3 * (df['income'] - df['income'].min()) / (df['income'].max() - df['income'].min()) +
            0.2 * (1 - df['debt_to_income_ratio']) +
            0.2 * (df['credit_history_length'] / df['credit_history_length'].max()) +
            0.15 * (1 - df['credit_utilization']) +
            0.1 * (df['employment_length'] / df['employment_length'].max()) +
            0.05 * (1 - df['num_late_payments'] / df['num_late_payments'].max()) +
            np.random.normal(0, 0.1, n_samples)
        )
        df['credit_quality'] = (credit_score > np.percentile(credit_score, 30)).astype(int)
        return df
    
    def feature_engineering(self, df):
        df = df.copy()
        df['income_per_year_employed'] = df['income'] / (df['employment_length'] + 1)
        df['debt_amount'] = df['income'] * df['debt_to_income_ratio']
        df['credit_utilization_category'] = pd.cut(df['credit_utilization'], 
                                                  bins=[0, 0.3, 0.7, 1.0], 
                                                  labels=['low', 'medium', 'high'])
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 30, 50, 100], 
                                labels=['young', 'middle', 'senior'])
        df['income_category'] = pd.cut(df['income'], 
                                      bins=[0, 50000, 100000, np.inf], 
                                      labels=['low', 'medium', 'high'])
        df['income_to_age_ratio'] = df['income'] / df['age']
        df['accounts_per_year'] = df['num_credit_accounts'] / (df['credit_history_length'] + 1)
        df['late_payment_rate'] = df['num_late_payments'] / (df['num_credit_accounts'] + 1)
        df['high_risk_indicators'] = (
            (df['debt_to_income_ratio'] > 0.4).astype(int) +
            (df['credit_utilization'] > 0.7).astype(int) +
            (df['num_late_payments'] > 2).astype(int) +
            (df['num_inquiries'] > 5).astype(int)
        )
        return df
    
    def preprocess_data(self, df, target_column='credit_quality'):
        df = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != target_column]
        if len(numeric_columns) > 0:
            imputer = SimpleImputer(strategy='median')
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        categorical_columns = [col for col in categorical_columns if col != target_column]
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        self.feature_names = X.columns.tolist()
        return X, y
    
    def train_models(self, X_train, y_train):
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
    
    def evaluate_models(self, X_test, y_test):
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        return results
    
    def plot_model_comparison(self, results):
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        model_names = list(results.keys())
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        for i, metric in enumerate(metrics):
            ax = axes[i//3, i%3]
            values = [results[model][metric] for model in model_names]
            bars = ax.bar(model_names, values)
            ax.set_title(f'{metric.upper()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        fig.delaxes(axes[1, 2])
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, results, y_test):
        plt.figure(figsize=(10, 8))
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            auc = result['roc_auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_feature_importance(self, model_name='Random Forest'):
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        model = self.models[model_name]
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} doesn't have feature importance!")
            return
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [self.feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
    
    def print_classification_reports(self, results, y_test):
        for name, result in results.items():
            print(f"\n{'='*50}")
            print(f"CLASSIFICATION REPORT - {name}")
            print(f"{'='*50}")
            print(classification_report(y_test, result['y_pred']))
            print(f"ROC-AUC Score: {result['roc_auc']:.4f}")
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='Random Forest'):
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
        elif model_name == 'Logistic Regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        print(f"Performing hyperparameter tuning for {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_

def main():
    credit_model = CreditScoringModel()
    print("Generating synthetic credit data...")
    df = credit_model.generate_synthetic_data(n_samples=10000)
    print("Performing feature engineering...")
    df = credit_model.feature_engineering(df)
    print("\nDataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Credit Quality Distribution:")
    print(df['credit_quality'].value_counts(normalize=True))
    print("\nPreprocessing data...")
    X, y = credit_model.preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_scaled = credit_model.scaler.fit_transform(X_train)
    X_test_scaled = credit_model.scaler.transform(X_test)
    print("\nTraining models...")
    credit_model.train_models(X_train_scaled, y_train)
    print("\nEvaluating models...")
    results = credit_model.evaluate_models(X_test_scaled, y_test)
    credit_model.print_classification_reports(results, y_test)
    print("\nGenerating visualizations...")
    credit_model.plot_model_comparison(results)
    credit_model.plot_roc_curves(results, y_test)
    credit_model.plot_feature_importance('Random Forest')
    print("\nPerforming hyperparameter tuning...")
    best_rf = credit_model.hyperparameter_tuning(X_train_scaled, y_train, 'Random Forest')
    if best_rf:
        y_pred_tuned = best_rf.predict(X_test_scaled)
        y_pred_proba_tuned = best_rf.predict_proba(X_test_scaled)[:, 1]
        print(f"\nTuned Random Forest Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred_tuned):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred_tuned):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred_tuned):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_tuned):.4f}")
    print("\nFeature Correlation Analysis:")
    plt.figure(figsize=(15, 12))
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    return credit_model, results

if __name__ == "__main__":
    model, results = main()
