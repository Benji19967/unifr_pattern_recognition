import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def plot_line(w, b):
    # Create x1 range
    x1 = np.linspace(-5, 5, 100)

    # Compute corresponding x2 values using: x2 = -(w1*x1 + b)/w2
    x2 = -(w[0] * x1 + b) / w[1]

    # Plot the hyperplane
    plt.plot(x1, x2, color="red")


def plot_data(X, y):
    # xi of category 0
    x1_0, x2_0 = X[y == 0, 0], X[y == 0, 1]
    # xi of category 1
    x1_1, x2_1 = X[y == 1, 0], X[y == 1, 1]

    plt.plot(x1_0, x2_0, "r^", x1_1, x2_1, "bs")  # (x1, y1, symbol_1, x2, y2, symbol_2)
    plt.plot()


def main():
    X, y = make_classification(n_features=2, n_redundant=0, random_state=0)

    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
    clf.fit(X, y)

    w = clf.named_steps["linearsvc"].coef_[0]
    b = clf.named_steps["linearsvc"].intercept_
    print(w)
    print(b)

    plot_data(X, y)
    plot_line(w, b)
    plt.show()


if __name__ == "__main__":
    main()
