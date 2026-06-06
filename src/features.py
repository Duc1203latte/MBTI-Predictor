import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, 'data', 'mbti_processed.csv')
    df = pd.read_csv(path)
    print(f"Đã load {len(df)} mẫu")
    return df

def extract_features(df):
    df['cleaned_posts'] = df['cleaned_posts'].fillna('')
    vectorizer = TfidfVectorizer(
        max_features=50000, #giữ lại 50000 từ phổ biến nhất
        ngram_range=(1, 2),
        min_df=2, #bỏ các từ chỉ xuất hiện ở đúng 1 người
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df['cleaned_posts'])
    print(f"Ma trận đặc trưng: {X.shape}")
    return X, vectorizer

def split_data(X, df):
    #Lấy 4 cột nhãn
    y = df[['I_E', 'N_S', 'T_F', 'J_P']]
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=0.2,
        random_state=42,
        stratify=df['type'] #giữ tỷ lệ của 16 nhóm
    )
    print(f"train {X_train.shape[0]} mẫu")
    print(f"test {X_test.shape[0]} mẫu")
    print(f"\n Phân phối nhóm MBTI trong tập train:")
    train_counts = df.loc[y_train.index, 'type'].value_counts()
    print(train_counts.to_string())

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    df = load_data()
    X, vectorizer = extract_features(df)

    print(f"\nSố người: {X.shape[0]}")
    print(f"Số đặc trưng (từ/cụm từ): {X.shape[1]}")

    vocab = vectorizer.get_feature_names_out()
    print(f"\n10 từ đầu trong vocabulary:\n{vocab[:10]}")

    first_row = X[0]
    df_view = pd.DataFrame({
        'từ': vocab[first_row.indices],
        'TF-IDF': first_row.data
    }).sort_values('TF-IDF', ascending=False).head(15)

    print(f"\nTop 15 từ đặc trưng nhất của người đầu tiên:")
    print(df_view.to_string(index=False))
    print(f"\nNhóm MBTI thực tế: {df['type'].iloc[0]}")

    X_train, X_test, y_train, y_test = split_data(X, df)

    # In bảng TF-IDF 10 dòng đầu, 10 từ có TF-IDF cao nhất
    vocab = vectorizer.get_feature_names_out()

    # Lấy 10 từ có TF-IDF trung bình cao nhất toàn dataset
    mean_tfidf = X.mean(axis=0).A1  # A1 chuyển matrix sang array 1D
    top10_indices = mean_tfidf.argsort()[::-1][:10]  # 10 chỉ số có giá trị lớn nhất
    top10_words = vocab[top10_indices]

    # Tạo dataframe chỉ gồm 10 cột đó, 10 dòng đầu
    import pandas as pd

    df_view = pd.DataFrame(
        X[:10, top10_indices].toarray(),
        columns=top10_words,
        index=df['type'].iloc[:10].values  # dùng nhãn MBTI làm tên hàng
    )

    print("\nBảng TF-IDF (10 người đầu × 10 từ đặc trưng nhất):")
    print(df_view.round(4).to_string())