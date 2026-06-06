import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

from src.train_svm import DIMENSIONS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import load_data, extract_features, split_data

DIMENSIONS = ['I_E', 'N_S', 'T_F', 'J_P']

def grid_search_rf(X_train, y_train, X_test, y_test, dim):
    print(f"Grid search RF - chiều {dim}")

    param_grid = [
        {'n_estimators': 100, 'max_depth': None, 'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_depth': 10, 'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_depth': 20, 'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_depth': None, 'max_features': 'log2'},
    ]

    best_f1 = 0
    best_model = None
    best_params = None
    results = []

    for params in param_grid:
        print(f"\n Thử max depth = {params['max_depth']}, max_features={params['max_features']}")
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            max_features=params['max_features'],
            class_weight='balanced',
            n_jobs = -1,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')

        #Thống kê dự đoán
        cm = confusion_matrix(y_test, y_pred)
        print(f" Macro F1: {f1:.4f}")
        print(f" Dự đoán đúng lớp 0: {cm[0][0]}/{cm[0][0]+cm[0][1]}")
        print(f" Dự đoán đúng lớp 1: {cm[1][1]}/{cm[1][0]+cm[1][1]}")

        results.append({**params, 'f1': f1})
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = params

    print(f"Kết quả tất cả tổ hợp - chiều {dim}")
    print("max_depth  max features  Macro F1")
    for r in results:
        marker = ' ← best' if r['max_depth'] == best_params['max_depth'] \
                                and r['max_features'] == best_params['max_features'] else ''
        print(f"{str(r['max_depth']) :<13} {r['max_features']:<16} {r['f1']:.4f}{marker}")

    print(f"\nBest: max_depth = {best_params['max_depth']}, "
          f"max_features = {best_params['max_features']},"
          f"F1={best_f1: .4f}")

    return best_model, best_params, best_f1

def train_all_dimensions(X_train, y_train, X_test, y_test):
    best_models = {}
    for dim in DIMENSIONS:
        y_tr = y_train[dim].values
        y_te = y_test[dim].values

        model, params, f1 = grid_search_rf(X_train, y_tr, X_test, y_te, dim)
        best_models[dim] = {
            'model': model,
            'params': params,
            'f1': f1
        }
    return best_models

def save_models(best_models):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    for dim, info in best_models.items():
        path = os.path.join(model_dir, f'rf_{dim}.pkl')
        joblib.dump(info['model'],path)
        print(f"Đã lưu: rf_{dim}.pkl")

if __name__ == '__main__':
    print("Load dữ liệu")
    df = load_data()
    X, vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X,df)
    best_models = train_all_dimensions(X_train, y_train, X_test, y_test)

    print("Tổng hợp kết quả")
    print("Chieu    max_depth    max_features    Macro F1")
    for dim, info in best_models.items():
        print(f"{dim:<8} {str(info['params']['max_depth']):<13} "
              f"{info['params']['max_features']:<16} {info['f1']:.4f}")

    save_models(best_models)