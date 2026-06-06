import os
import sys
import joblib
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import load_data, extract_features, split_data

DIMENSIONS  = ['I_E', 'N_S', 'T_F', 'J_P']
ALGORITHMS  = ['lr', 'svm', 'rf']

def load_all_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'models')

    models= {}
    for algo in ALGORITHMS:
        for dim in DIMENSIONS:
            path = os.path.join(model_dir,f'{algo}_{dim}.pkl')
            models[f'{algo}_{dim}'] = joblib.load(path)
    return models

def evaluate_all(models, X_test, y_test):
    #Lưu F1 của từng thuật toán cho từng chiều
    scores = {dim: {} for dim in DIMENSIONS}
    for dim in DIMENSIONS:
        y_true = y_test[dim].values
        for algo in ALGORITHMS:
            model = models[f'{algo}_{dim}']
            y_pred = model.predict(X_test)
            f1 = f1_score(y_true, y_pred, average = 'macro')
            scores[dim][algo] = f1

    return scores

def select_best(scores):
    best = {}
    for dim in DIMENSIONS:
        #Tìm thuật toán c f1 cao nhất cho chiều này
        best_algo = max(scores[dim], key = scores[dim].get)
        best[dim] = {
            'algo': best_algo,
            'f1': scores[dim][best_algo]
        }
    return best

def save_best_models(models, best):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    for dim, info in best.items():
        algo = info['algo']
        src_model = models[f'{algo}_{dim}']
        path = os.path.join(models_dir, f'best_{dim}.pkl')
        joblib.dump(src_model, path)
        print(f"Đã lữu best_{dim}.pkl (thuật toán: {algo})")

def print_comparison(scores, best):
    print("Bảng so sánh macro F1")
    print(f"{'Chiều':<8} {'LR':<10} {'SVM':<10} {'RF':<10} {'Best':<12}")
    for dim in DIMENSIONS:
        lr = scores[dim]['lr']
        svm = scores[dim]['svm']
        rf = scores[dim]['rf']
        best_algo = best[dim]['algo'].upper()

        # Đánh dấu * vào giá trị tốt nhất
        lr_s = f"{lr:.4f}" + ('*' if best[dim]['algo'] == 'lr' else ' ')
        svm_s = f"{svm:.4f}" + ('*' if best[dim]['algo'] == 'svm' else ' ')
        rf_s = f"{rf:.4f}" + ('*' if best[dim]['algo'] == 'rf' else ' ')

        print(f"{dim:<8} {lr_s:<10} {svm_s:<10} {rf_s:<10} {best_algo:<12}")

        # F1 trung bình của bộ best
    avg_f1 = sum(best[dim]['f1'] for dim in DIMENSIONS) / len(DIMENSIONS)
    print("─" * 55)
    print(f"Macro F1 trung bình của bộ model tốt nhất: {avg_f1:.4f}")

if __name__ == '__main__':
    print("Load dữ liệu...")
    df = load_data()
    X, vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X, df)

    print("Load 12 model đã train...")
    models = load_all_models()

    print("Đánh giá tất cả model trên tập test...")
    scores = evaluate_all(models, X_test, y_test)

    best = select_best(scores)
    print_comparison(scores, best)

    save_best_models(models, best)

    # Lưu cả vectorizer để dùng trong API
    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vec_path = os.path.join(base_dir, 'models', 'vectorizer.pkl')
    joblib.dump(vectorizer, vec_path)
    print(f"\nĐã lưu vectorizer.pkl")

