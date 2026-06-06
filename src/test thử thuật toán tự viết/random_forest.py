import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,value=None):
        self.feature = feature #index của feature dùng để chia
        self.threshold = threshold #ngưỡng để chia (Tf-IDF > threshold)
        self.left = left
        self.right = right
        self.value = value #chỉ có giá trị khi là node lá

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth #độ saau tối đa của cây
        self.min_samples_split = min_samples_split #node caanf ít nhất bao nhiêu mẫu ới chia hết
        self.max_features = max_features #số feature xét tại mỗi node
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y,depth=0)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        #Điều kiện dừng
        if (depth >= self.max_depth
                or n_classes == 1
                or n_samples < self.min_samples_split):
            return Node(value=self._most_common(y))

        #Chọn ngẫu nhiên max_features, feature để xét
        n_feats = self.max_features or n_features
        feature_indices = np.random.choice(n_features, n_feats, replace=False)

        #Tìm điểm chia tốt nhâất
        best_feature, best_threshold = self._best_split(X,y,feature_indices)

        #Chia dữ liệu
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        #Đệ quy xy 2 nhánh con
        left = self._grow_tree(X[left_mask], y[left_mask], depth+1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth+1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feature_indices):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in feature_indices:
            threshold = np.unique(X[:, feature])
            for threshold in threshold:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                #Bỏ qua nếu một bên rỗng
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                gini = self._weighted_gini(
                    y[left_mask], y[right_mask], len(y)
                )

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _weighted_gini(self, y_left, y_right, n_total):
        gini_left = self._gini(y_left)
        gini_right = self._gini(y_right)
        return (len(y_left) / n_total) * gini_left \
            + (len(y_right) / n_total) * gini_right

    def _gini(self,y):
        #Gini = 1 - Σ(p_c²)
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _most_common(self,y):
        #Trả về nhãn xuất hiện nhiều nhất
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        #Node lá trả về nhãn
        if node.value is not None:
            return node.value
        #Node trong thì rẽ trái hoặc phải
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

class RandomForest:
    def __init__(self,n_trees=100, max_depth=10, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = [] #danh sách cây đã train

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            #Lấy mẫu có hoàn lại
            indices = np.random.choice(len(y), len(y), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            #Train 1 cây trên tập bootstrap
            tree = DecisionTree(
                max_depth = self.max_depth,
                min_samples_split = self.min_samples_split,
                max_features = self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            if (i+1) % 10 == 0:
                print(f"Đã train {i+1}/{self.n_trees} cây")

    def predict(self, X):
        #Lấy dự đoán của từng cây
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        #alll_preds shape: (n_trees, n_samples)
        #Bỏ phiếu đa ố cho từng mẫu
        return np.array([
            np.bincount(all_preds[:,i]).argmax()
            for i in range(X.shape[0])
        ])

if __name__ == '__main__':
    import os
    import sys
    from scipy.sparse import issparse

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from features import load_data, extract_features, split_data

    print("=" * 40)
    print("Load dữ liệu...")
    print("=" * 40)
    df = load_data()
    X, vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X, df)

    # Chuyển sang dense array
    print("\nChuyển sang dense array...")
    X_train_dense = X_train.toarray() if issparse(X_train) else X_train
    X_test_dense  = X_test.toarray()  if issparse(X_test)  else X_test

    # Test với chiều I_E
    print("Train Random Forest — chiều I/E")
    y_train_ie = y_train['I_E'].values
    y_test_ie  = y_test['I_E'].values

    model = RandomForest(
        n_trees=10,
        max_depth=5,
        max_features=224   # √50000
    )
    model.fit(X_train_dense, y_train_ie)

    # Đánh giá
    print("\nĐang dự đoán tập test...")
    y_pred = model.predict(X_test_dense)
    accuracy = np.mean(y_pred == y_test_ie)
    print(f"\nAccuracy: {accuracy:.4f}")

    print(f"Model dự đoán I: {np.sum(y_pred == 1)}")
    print(f"Model dự đoán E: {np.sum(y_pred == 0)}")
    print(f"Thực tế I:       {np.sum(y_test_ie == 1)}")
    print(f"Thực tế E:       {np.sum(y_test_ie == 0)}")


