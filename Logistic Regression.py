import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("loan-train.csv")
data.dropna(axis=0, inplace=True)

data['Dependents'] = data['Dependents'].replace({'3+': 3,'1':1,'2':2,'0':0})
data['Gender'] = data['Gender'].replace({'Male':1,'Female':0})
data['Married'] = data['Married'].replace({'Yes':1,'No':0})
data['Education'] = data['Education'].replace({'Graduate':1,'Not Graduate':0})
data['Self_Employed'] = data['Self_Employed'].replace({'Yes':1,'No':0})
data['Property_Area'] = data['Property_Area'].replace({'Urban':1,'Rural':0,'Semiurban':0})
data['Loan_Status'] = data['Loan_Status'].replace({'Y':1,'N':0})

X = data.drop(columns=['Loan_Status', 'Loan_ID'])
y = data['Loan_Status']

X = (X - X.mean()) / X.std()

X = np.array(X)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient_descent(X, y, learning_rate, iterations):
    m, n = X.shape
    theta = np.zeros(n)
    b = 0
    costs = []
    for _ in range(iterations):
        y_pred = sigmoid(np.dot(X, theta)+b)
        error = y_pred - y
        gradient = np.dot(X.T, error) / m
        theta -= learning_rate * gradient
        b -= learning_rate*error/m
        cost = -1 / m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        costs.append(cost)

    return theta, costs

learning_rate = 0.05
iterations = 50000

theta, costs = gradient_descent(X, y, learning_rate, iterations)

plt.plot(range(1, iterations + 1), costs)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.title('Learning Curve')
plt.show()

print(costs[-1])