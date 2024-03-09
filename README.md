# Implementation-of-Linear-Regression-Using-Gradient-Descent
### Name : Anbuselvan.S
### Reference No: 212223240008

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Start the program
2. Import numpy as np 3.Plot the points
3. IntiLiaze thhe program
4. End the program

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Anbuselvan.S
RegisterNumber:212223240008  
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros(X.shape[1]).reshape(-1, 1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1, 1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
    return theta

data = pd.read_csv("/content/50_Startups.csv")

X = (data.iloc[1:, :-2].values)
X1=X.astype(float)
scaler = StandardScaler()

y = (data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)

theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot (np.append(1, new_Scaled), theta)

prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)

print(f"Predicted value: {pre}")
```

## Output:
### Dataset Given:
![Screenshot 2024-03-09 091711](https://github.com/anbuselvan1519/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/139841744/ba160574-32e4-4b6f-b555-d4f5da92814d)

### Predicted Value:
![Screenshot 2024-03-09 091628](https://github.com/anbuselvan1519/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/139841744/19e133f1-81bf-487e-98ee-b04cafe63513)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
