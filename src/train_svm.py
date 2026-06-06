import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import load_data, extract_features, split_data

DIMENSIONS = ['I_E', 'N_S', 'T_F', 'J_P']

def grid_search_svm(X_train, y_train, X_test, y_test, dim):
    print(f"Grid search SVM - chieeuf {dim}")
    param_grid = [
        {'kernel': 'linear', 'C': 0.01},
        {'kernel': 'linear', 'C': 0.1},
        {'kernel': 'linear', 'C': 1.0},
        {'kernel': 'linear', 'C': 10.0},
    ]

    best_f1 = 0
    best_model = None
    best_params = None
    results = []

    for params in param_grid:
        print(f"\n Thu: kernel = {params['kernel']}, C={params['C']}")
        if params['kernel'] =='linear':
            model = LinearSVC(
                C=params['C'],
                class_weight='balanced',
                max_iter=1000
            )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f" Macro f1: {f1:.4f}")
        # Thống kê dự đoán
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Dự đoán đúng lớp 0: {cm[0][0]}/{cm[0][0] + cm[0][1]}")
        print(f"  Dự đoán đúng lớp 1: {cm[1][1]}/{cm[1][0] + cm[1][1]}")
        results.append({**params, 'f1':f1})

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = params

    print(f"Ket qua to hop - chieu {dim}:")
    print("kernel     C        Macro F1")
    for r in results:
        marker = '← best' if r == {**best_params, 'f1':best_f1} else ''
        print(f"{r['kernel']:<10} {r['C']:<8} {r['f1']:.4f} {marker}")
    print(f"\nBest: kernel={best_params['kernel']}, C={best_params['C']}, F1={best_f1:.4f}")
    return best_model, best_params, best_f1

def train_all_dimensions(X_train, X_test, y_train, y_test):
    best_models = {}
    for dim in DIMENSIONS:
        y_tr = y_train[dim].values
        y_te = y_test[dim].values
        model, params, f1 = grid_search_svm(
            X_train, y_tr, X_test, y_te, dim
        )
        best_models[dim] = {
            'model': model,
            'params': params,
            'f1':f1
        }
    return best_models

def save_models(best_models):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    for dim, info in best_models.items():
        path = os.path.join(models_dir, f'svm_{dim}.pkl')
        joblib.dump(info['model'], path)
        print(f"Da luwu: svm_{dim}.pkl")

if __name__ == '__main__':
    print("load data...")
    df = load_data()
    X, vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X,df)

    best_models = train_all_dimensions(X_train, X_test, y_train, y_test)
    print("Tong ho kq SVM")
    print(f"{'Chiều':<8} {'Kernel':<10} {'C':<8} {'Macro F1'}")
    for dim, info in best_models.items():
        print(f"{dim:<8} {info['params']['kernel']:<10} {info['params']['C']:<8} {info['f1']:.4f}")

    save_models(best_models)