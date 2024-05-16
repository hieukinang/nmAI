from collections import defaultdict


class Bayes:
    def __init__(self):
        self.class_probabilities = defaultdict(float)
        self.feature_probabilities = defaultdict(lambda: defaultdict(float))

    def fit(self, X_train, y_train):
        total_samples = len(y_train)
        # Tính toán xác suất của các nhãn
        for label in set(y_train):#lọc dữ liệu duy nhất
            label_samples = [X_train[i] for i in range(total_samples) if y_train[i] == label]#lọc các mẫu có cùng nhãn
            self.class_probabilities[label] = len(label_samples) / total_samples
            # Tính toán xác suất của các đặc trưng dựa trên nhãn
            for i in range(len(X_train[0])):
                feature_values = [sample[i] for sample in label_samples]
                total_values = len(feature_values)
                unique_values = set(feature_values)
                for value in unique_values:
                    self.feature_probabilities[label][(i, value)] = feature_values.count(value) / total_values

    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            max_probability = -1
            predicted_label = None
            # Dự đoán nhãn cho mỗi mẫu dựa trên xác suất Bayes
            for label in self.class_probabilities:
                probability = self.class_probabilities[label]
                for i in range(len(sample)):
                    probability *= self.feature_probabilities[label][(i, sample[i])]
                if probability > max_probability:
                    max_probability = probability
                    predicted_label = label
            predictions.append(predicted_label)
        return predictions


# Dữ liệu mẫu giả định
X_train = [
    ['Strong', 'Sunny'],
    ['Weak', 'Sunny'],
    ['Weak', 'Rainy'],
    ['Strong', 'Rainy'],
    ['Strong', 'Overcast'],
    ['Strong', 'Overcast'],
    ['Strong', 'Sunny'],
    ['Strong', 'Sunny']
]
y_train = ['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']

X_test = [
    ['Weak', 'Sunny'],
    ['Strong', 'Rainy'],
    ['Strong', 'Overcast']
]

# Tạo và huấn luyện mô hình
model = Bayes()
model.fit(X_train, y_train)

# Dự đoán nhãn cho dữ liệu kiểm tra
predictions = model.predict(X_test)
print("Predictions:", predictions)
