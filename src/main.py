from features import load_data, extract_features, split_data

def main():
    print("=" * 40)
    print("BƯỚC 1: Load và trích xuất đặc trưng")
    print("=" * 40)
    df = load_data()
    X, vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X, df)

    # Sau này thêm tiếp:
    # print("BƯỚC 2: Train model")
    # models = train_all(X_train, y_train)
    #
    # print("BƯỚC 3: Đánh giá")
    # evaluate_all(models, X_test, y_test)

if __name__ == '__main__':
    main()