import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iteractions=1000, class_weight = None):
        self.learning_rate = learning_rate
        self.n_iteractions = n_iteractions
        self.class_weight = class_weight #Thêm trọng số phạt
        self.weights = None
        self.bias = None

    def _compute_sample_weights(self, y):
        n_samples = len(y)
        classes = np.unique(y)
        weights = np.ones(n_samples)  # mặc định tất cả = 1

        if self.class_weight == 'balanced':
            for c in classes:
                n_c = np.sum(y == c)
                w_c = n_samples / (len(classes) * n_c)
                weights[y == c] = w_c  # gán trọng số cho từng mẫu

        return weights

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        #Khởi tạo weght = 0, bias = 0
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        y = np.array(y)
        sample_weights = self._compute_sample_weights(y)
        for i in range(self.n_iteractions):
            #Tính score
            z = X.dot(self.weights) + self.bias
            #Đưa qua sigmoid để tính xác suất
            y_predicted = self._sigmoid(z)

            #Nhân sai số với sample_weight trước khi tính gradient
            error = y_predicted - y
            #Tính gradient
            dw = (1/n_samples) * X.T.dot(sample_weights*error)
            db = (1/n_samples) * np.sum(sample_weights*error)
            #Cập nật weight và bís
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            #In loss mỗi 100 vòng để theo dõi
            if i % 100 == 0:
                loss = self._compute_loss(y, y_predicted)
                print(f" Iteration {i} Loss: {loss:.4f}")

    def _compute_loss(self, y, y_predicted, sample_weights=None):
        #Clip ể tránh log(0)
        y_predicted = np.clip(y_predicted,1e-7, 1 - 1e-7 )
        loss = -(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
        if sample_weights is not None:
            return np.mean(sample_weights * loss)  # có trọng số
        return np.mean(loss)  # không có trọng số

    def predict_proba(self,X):
        #Trả về xác suất
        z = X.dot(self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X):
        #Xác suất >= 0,5 thì l lớp 1, còn lại lớp 0
        return (self.predict_proba(X) >= 0.5).astype(int)

if __name__ == "__main__":
    import os
    import sys
    import numpy as np
    from scipy.sparse import issparse

    #Thêm thư mục srrc vào path để import được feature
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from features import load_data, extract_features, split_data

    print("=" * 40)
    print("Load dữ liệu")
    df = load_data()
    X, vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X,df)

    #Chuyển ma trận sang array thường
    X_train_dense = X_train.toarray() if issparse(X_train) else X_train
    X_test_dense = X_test.toarray() if issparse(X_test) else X_test

    #Test với I_E
    print("Train với chiều I/E")
    y_train_ie = y_train['I_E'].values
    y_test_ie = y_test['I_E'].values
    model = LogisticRegression(learning_rate=0.1, n_iteractions=500, class_weight='balanced')
    model.fit(X_train_dense, y_train_ie)

    #Đánh giá
    y_pred = model.predict(X_test_dense)
    accuracy = np.mean(y_pred == y_test_ie)
    print(f"\nAccuracy trên tập test: {accuracy:.4f}")

    #Phân phối dự đoán
    print(f"Model dự đoán I: {np.sum(y_pred == 1)}")
    print(f"Model dự đoán E: {np.sum(y_pred == 0)}")
    print(f"Thực tế I: {np.sum(y_test_ie == 1)}")
    print(f"Thực tế E: {np.sum(y_test_ie == 0)}")