# 🧠 MBTI Personality Predictor

Dự đoán nhóm tính cách MBTI từ văn bản sử dụng Machine Learning.

## Giới thiệu

Ứng dụng dự đoán nhóm tính cách MBTI (Myers-Briggs Type Indicator) 
dựa trên văn bản người dùng nhập vào. Thay vì phân loại 16 nhóm cùng lúc, 
bài toán được chia thành 4 bài toán nhị phân độc lập:

- **I/E** — Introvert vs Extrovert
- **N/S** — Intuition vs Sensing  
- **T/F** — Thinking vs Feeling
- **J/P** — Judging vs Perceiving

## Dataset

- Nguồn: [MBTI 500 — Kaggle](https://www.kaggle.com/datasets/datasnaek/mbti-type)
- 8675 mẫu, mỗi mẫu gồm 50 bài đăng ghép lại

## Các thuật toán sử dụng

| Thuật toán | Thư viện | Macro F1 trung bình |
|---|---|---|
| Logistic Regression | scikit-learn | ~0.70 |
| SVM (LinearSVC) | scikit-learn | ~0.70 |
| Random Forest | scikit-learn | ~0.57 |

Kết quả best model cho từng chiều:

| Chiều | Best model | Macro F1 |
|---|---|---|
| I/E | Logistic Regression | 0.6721 |
| N/S | SVM | 0.6721 |
| T/F | Logistic Regression | 0.7995 |
| J/P | SVM | 0.6597 |

## Cấu trúc project
## Cấu trúc project

- 📁 **data/** — dữ liệu
  - `mbti_1.csv` — dataset gốc (không commit)
- 📁 **models/** — các model đã train
  - `best_I_E.pkl` — model tốt nhất chiều I/E
  - `best_N_S.pkl` — model tốt nhất chiều N/S
  - `best_T_F.pkl` — model tốt nhất chiều T/F
  - `best_J_P.pkl` — model tốt nhất chiều J/P
  - `vectorizer.pkl` — TF-IDF vectorizer
- 📁 **src/** — source code
  - `preprocess.py` — tiền xử lý dữ liệu
  - `features.py` — TF-IDF + train/test split
  - `train_lr_sklearn.py` — train Logistic Regression
  - `train_svm.py` — train SVM
  - `train_rf.py` — train Random Forest
  - `select_best.py` — chọn best model
  - `predict.py` — hàm dự đoán
- `app.py` — Streamlit web app
- `requirements.txt`
- `README.md`
## Cài đặt và chạy local

```bash
# Clone repo
git clone https://github.com/Duc1203latte/MBTI-Predictor.git
cd mbti-predictor

# Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Cài thư viện
pip install -r requirements.txt

# Chạy app
streamlit run app.py
```

## Demo

Truy cập: [https://mbti-predictor-wcxkutnyb22kyqdcduzhv8.streamlit.app/]

## Tác giả

- **Họ tên:** Đỗ Hồng Đức
- **Đơn vị:** Đại học BKHN
- **MSSV:** 20235040