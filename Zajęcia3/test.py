import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''# Dane

# Przykładowa macierz X
X = np.array([[2], [1]])  # Przykładowa macierz X

# Stworzenie kolumny wypełnionej jedynkami
ones_column = np.ones((X.shape[0], 1))  # Tworzenie kolumny jedynkowej o takiej samej liczbie wierszy jak X

# Połączenie kolumny jedynkowej z macierzą X
X = np.hstack((ones_column, X))
X = np.matrix(X)
Y = np.array([[10], [12]])  # Przykładowe dane Y
Y = np.hstack((ones_column, Y))
Y = np.matrix(Y)

print(X)
print(X.T)
print(X @ X.T)

print(np.linalg.pinv(np.dot(X, X.T)))
print(X @ Y.T)

_T = np.linalg.pinv(np.dot(X, X.T)) @ X @ Y.T

def model(parametry, x):
    a, b = parametry
    return a * x + b

def plot_fig(X: np.ndarray, Y: np.ndarray, coeff: np.ndarray):
    X_test = np.linspace(start=X.min(), stop=X.max(), num=300)  
    func_str = "y = "
    Y_pred = model(coeff, X_test)
    for i, c in enumerate(coeff.ravel()[::-1]):
        func_str += f"{round(c, 4)} * x ** {i} + "

    plt.scatter(X, Y, label='dane rzeczywiste')
    plt.plot(X_test, Y_pred, color='tab:orange', label='estymowany trend')
    plt.xlabel('x - PKB na osobę', fontsize=14)
    plt.ylabel('y - poczucie szczęścia', fontsize=14)
    plt.title(f"Dopasowano funkcję: {func_str[:-2]}")
    plt.legend()
    plt.show()

plot_fig(X, Y, _T)
'''
# Importing Necessary Libraries
 
plt.rcParams['figure.figsize'] = (20.0, 10.0)
 
# Reading Data
data = pd.read_csv('headbrain.csv')
print(data.shape)
data.head()
 
 
# Collecting X and Y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

# Plotting Values and Regression Line
max_x = np.max(X) + 100
min_x = np.min(X) - 100
# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = c + m * x 
 
# Ploting Line
plt.plot(x, y, color='#52b920', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef4423', label='Scatter Plot')
 
plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()