import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, n_iterations=1000, C=1.0, class_weight=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.C = C #hệ số cân bằng margin và loss
        self.class_weight = class_weight
        self.weights = None
        self.bias = None

    def _compute_sample_weights(self, y):
        n_samples = len(y)
        classes = np.unique(y)
        weights = np.ones(n_samples)

        if self.class_weight == 'balanced':
            for c in classes:
                n_c = np.sum(y == c)
                w_c = n_samples / (len(classes) * n_c)
                weights[y == c] = w_c

        return weights

    def _convert_labels(self, y):
        return np.where(y==1,1,-1)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = np.zeros(n_features)
        self.bias = 0.0

        #Chuyển nhãn sang +1/-1
        y_svm = self._convert_labels(y)
        sample_weights = self._compute_sample_weights(y)
        for i in range(self.n_iterations):
            #Tính score cho tất cả các mẫu
            scores = X.dot(self.weights) + self.bias
            #Tìm các mẫu vi phạm margin
            margins = y_svm * scores
            violated = (margins < 1).astype(float) #True/False array

            #Vector hóa
            mask = violated * sample_weights * y_svm
            dw = self.weights - self.C * X.T.dot(mask)
            db = -self.C * np.sum(mask)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                loss = self._compute_loss(scores, y_svm, sample_weights)
                print(f" Iteration: {i}, Loss: {loss:.4f}")

    def _compute_loss(self, scores, y_svm, sample_weights):
        #Hinge loss có trọng số
        hinge = np.maximum(0,1- y_svm * scores)
        regularization = 0.5 * np.dot(self.weights, self.weights)
        return regularization + self.C * np.mean(sample_weights * hinge)

    def predict(self, X):
        scores = X.dot(self.weights) + self.bias
        #Score > 0 -> trả về 1
        #Score < 0 -> trả về 0
        return (scores >= 0).astype(int)

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
    model = SVM(learning_rate=0.0001, n_iterations=500, C=1.0,  class_weight='balanced')
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