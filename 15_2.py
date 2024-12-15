import numpy as np

# Функция активации (шаговая функция)
def step_function(x):
    return np.where(x >= 0, 1, 0)

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.W = np.zeros(input_size + 1)  # Инициализация весов (включая bias)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, x):
        return step_function(np.dot(self.W, x))

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1)  # Вставка смещения (bias)
                prediction = self.predict(xi)
                self.W += self.learning_rate * (target - prediction) * xi

# Данные для обучения (И, ИЛИ)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])  # Операция И (AND)
y_or = np.array([0, 1, 1, 1])   # Операция ИЛИ (OR)

# Список эпох для тестирования
epochs_list = [10000, 20000, 50000]

for epochs in epochs_list:
    print(f"\nTraining Perceptron with {epochs} epochs for AND operation:")
    perceptron = Perceptron(input_size=2, epochs=epochs)
    perceptron.train(X, y_and)

    # Тестирование
    for xi in X:
        xi_with_bias = np.insert(xi, 0, 1)  # Вставка смещения (bias) для тестирования
        print(f"{xi} -> {perceptron.predict(xi_with_bias)}")

    print(f"\nTraining Perceptron with {epochs} epochs for OR operation:")
    perceptron = Perceptron(input_size=2, epochs=epochs)
    perceptron.train(X, y_or)

    # Тестирование
    for xi in X:
        xi_with_bias = np.insert(xi, 0, 1)  # Вставка смещения (bias) для тестирования
        print(f"{xi} -> {perceptron.predict(xi_with_bias)}")

