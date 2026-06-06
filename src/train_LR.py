import os
import sys
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix

from src.train_svm import DIMENSIONS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import load_data, extract_features, split_data
import warnings
warnings.filterwarnings('ignore')

DIMENSIONS = ['I_E', 'N_S', 'T_F', 'J_P']

def grid_search_lr(X_train, y_train, X_test, y_test, dim):
    print(f" Grid search LR - chiều {dim}")

    param_grid = [
        {'C': 0.1, 'penalty': 'l2'},
        {'C': 1.0, 'penalty': 'l2'},
        {'C': 10.0, 'penalty': 'l2'},
        {'C': 1.0, 'penalty': 'l1'},
        {'C': 10.0, 'penalty': 'l1'},
    ]
    best_f1 = 0
    best_model = None
    best_params = None
    results = []

    for params in param_grid:
        print(f"Thử: C={params['C']}, penalty = {params['penalty']}")
        model = LogisticRegression(
            C=params['C'],
            penalty=params['penalty'],
            solver='liblinear',  # hỗ trợ cả l1 và l2
            class_weight='balanced',
            max_iter=8000
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Macro F1: {f1:.4f}")
        print(f"  Dự đoán đúng lớp 0: {cm[0][0]}/{cm[0][0] + cm[0][1]}")
        print(f"  Dự đoán đúng lớp 1: {cm[1][1]}/{cm[1][0] + cm[1][1]}")
        results.append({**params, 'f1': f1})
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = params

    print(f"Kết quả các tổ hợp - chều {dim}:")
    print("C         penalty      Macro f1")

    for r in results:
        marker = ' ← best' if r['C'] == best_params['C'] \
                              and r['penalty'] == best_params['penalty'] else ''
        print(f"{str(r['C']):<8} {r['penalty']:<10} {r['f1']:.4f}{marker}")

    print(f"\nBest: C={best_params['C']}, "
          f"penalty={best_params['penalty']}, F1={best_f1:.4f}")

    return best_model, best_params, best_f1

def train_all_dimensions(X_train, X_test, y_train, y_test):
    best_model = {}
    for dim in DIMENSIONS:
        y_tr = y_train[dim].values
        y_te = y_test[dim].values

        model, params, f1 = grid_search_lr(
            X_train, y_tr, X_test, y_te, dim
        )
        best_model[dim] = {
            'model': model,
            'params': params,
            'f1': f1
        }
    return best_model

def save_model(best_models):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    for dim, info in best_models.items():
        path = os.path.join(models_dir, f'lr_{dim}.pkl')
        joblib.dump(info['model'], path)
        print(f"đã lưu lr_{dim}.pkl")

if __name__ == '__main__':
    print("Load dữ liệu...")
    df = load_data()
    X, vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X, df)

    best_models = train_all_dimensions(X_train, X_test, y_train, y_test)

    # Tổng hợp kết quả
    print("TỔNG HỢP KẾT QUẢ LOGISTIC REGRESSION")
    print("Chieu    C        penalty    Macro F1")
    print("─" * 45)
    for dim, info in best_models.items():
        print(f"{dim:<8} {str(info['params']['C']):<8} "
              f"{info['params']['penalty']:<10} {info['f1']:.4f}")

    save_model(best_models)