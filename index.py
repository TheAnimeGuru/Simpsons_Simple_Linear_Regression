import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Hyper Params ###

learning_rate = 0.015
iters = 10000
w = 0
b = 0

## Get Data

path = './data/ViewsByTitle.csv'
df = pd.read_csv(path)

## x = views, y = ImDB rating
x = np.array(pd.to_numeric(df.imdb_rating))
y = np.array(pd.to_numeric(df.views_title))

def jwb(x, y, w, b):
    """
    Calculates squared loss of Function Fwb(x) => J(w,b)

     Args:
      x (ndarray (m,))  : Data x, m len
      y (ndarray (m,))  : Real y, m len
      w (scalar): Initial model weight
      b (scalar): initial model bias
      
    Returns:
      cost (float) : Squared Cost
    """
    cost = 0
    m = x.shape[0]

    for i in range(m-1):
        f_wb = w*x[i] + b
        cost += (f_wb - y[i])**2
    
    return cost / (2*m)

def dw_jwb(x, y, w, b):
    """
    Calculates cost of partial derivative w of J(w,b)

    Args:
      x (ndarray (m,))  : Data x, m len
      y (ndarray (m,))  : Real y, m len
      w (scalar): Initial model weight
      b (scalar): initial model bias
      
    Returns:
      cost (float) : Cost of partial derivative w of J(w,b)
    """
    cost = 0
    m = x.shape[0]

    for i in range(m-1):
        f_wb = w*x[i] + b
        cost += (f_wb - y[i]) * x[i]
    
    cost = cost / m
    return cost

def db_jwb(x, y, w, b):
    """
    Calculates cost of partial derivative b of J(w,b)

    Args:
      x (ndarray (m,))  : Data x, m len
      y (ndarray (m,))  : Real y, m len
      w (scalar): Initial model weight
      b (scalar): initial model bias
      
    Returns:
      cost (float) : Cost of partial derivative b of J(w,b)
    
    """
    cost = 0
    m = x.shape[0]

    for i in range(m-1):
        f_wb = w*x[i] + b
        cost += f_wb - y[i]
    
    cost = cost / m
    return cost

def gradient_descent(x, y, w, b, a):
    """
    Executes Gradient Descent and returns updated values model parameters
    
    Args:
      x (ndarray (m,))  : Data x, m len
      y (ndarray (m,))  : Real y, m len
      w (scalar): Initial model weight
      b (scalar): initial model bias
      alpha (float): Learning rate
      
    Returns:
      w (scalar): Updated weight
      b (scalar): Updated bias
    """
    # Use temp variables to execute parallel update
    w_new = w - a * dw_jwb(x, y, w, b)
    b_new = b - a * db_jwb(x, y, w, b)

    return (w_new, b_new)

def predict(x, w, b):
    """
    Returns y_hat of linear model

     Args:
      x (ndarray (m,))  : Data x, m len
      w (scalar): Initial model weight
      b (scalar): initial model bias
      
    Returns:
      y_hat (float): Prediction of y for datapoint x
    """
    return w*x + b

if __name__ == '__main__':
    print('Starting training...')
    for iter in range(iters):
        w, b = gradient_descent(x, y, w, b, learning_rate)
        if iter % 100 == 0 or iter == iters-1:
            cost = jwb(x, y, w, b)
            print(f'Iteration: {iter}, Weight: {w:.4f}, Bias: {b:.4f}, Cost: {cost:.4f}')
    fig, ax = plt.subplots()
    ax.set_xlabel('IMDB Rating')
    ax.set_ylabel('Views')
    prediction, = ax.plot([5, 10], [predict(5, w, b), predict(10, w, b)], label="Prediction")
    training_data = ax.scatter(x, y, marker="x", color="red", label="Actual Data")
    ax.legend(handles=[training_data, prediction])
    plt.show()
