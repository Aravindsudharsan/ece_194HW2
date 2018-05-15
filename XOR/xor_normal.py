import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


# Decision region plotting
def plot_decision_regions(X, y, predict, weight0, weight1, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = predict(np.array([xx1.ravel(), xx2.ravel()]).T, weight0, weight1)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for i in range(len(X)):
        if y[i] == -1:
            plt.scatter(X[i][0],X[i][1], alpha=0.8, c = cmap(0), marker=markers[0], label=-1)
        elif y[i] == 1:
            plt.scatter(X[i][0],X[i][1], alpha=0.8, c = cmap(1), marker=markers[1], label=1)
    

# Original input
def plot_original_input(X, y):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    for i in range(len(X)):
        if y[i] == -1:
            plt.scatter(X[i][0],X[i][1], alpha=0.8, c = cmap(0), marker=markers[0], label=-1)
        elif y[i] == 1:
            plt.scatter(X[i][0],X[i][1], alpha=0.8, c = cmap(1), marker=markers[1], label=1)

# Loss function
def loss_function(y,y_cap,examples):
    # Mean Square error
    error = y - y_cap
    loss = ((np.sum(np.power(error,2)))/(2*examples))
    return (loss)

# Derivative of loss function
def derrivative_loss_fn(y,y_cap,examples):
    return (y - y_cap)/(examples)

# activation function
def tanh_act(x):
    return np.tanh(x)

# Derivative of activation function
def derivate_tanh_act(x):
    return (1-x*x)

# Forward propagation outputs
def forward_propagate(x,weight0,weight1):
    # Layer 1
    ones = np.ones(np.shape(x)[0])
    layer1_input = np.column_stack((ones, x))
    layer1_sum = np.dot(layer1_input, weight0.T)
    layer1_output = tanh_act(layer1_sum)

    # Layer 2
    ones = np.ones(np.shape(layer1_output)[0])
    layer2_input = np.column_stack((ones, layer1_output))
    layer2_sum = np.dot(layer2_input, weight1.T)
    layer2_output = tanh_act(layer2_sum)

    return layer1_input, layer1_sum, layer1_output, layer2_input, layer2_sum, layer2_output

# Prediction function
def predict(x,weight0,weight1):
    layer1_in, layer1_sum, layer1_out, layer2_in, layer2_sum, y_cap = forward_propagate(x,weight0,weight1)
    return y_cap


# Back propagation for updating weights
def back_propagate(x,y,weight0,weight1,learning_rate,epochs):
    loss = []
    examples = np.shape(x)[0]
    for j in range(epochs):
        layer1_in, layer1_sum, layer1_out, layer2_in, layer2_sum, y_cap = forward_propagate(x,weight0,weight1)
        loss.append(loss_function(y,y_cap,examples))
        derivative_loss = derrivative_loss_fn(y,y_cap,examples)
        layer2_error = derivative_loss * derivate_tanh_act(y_cap)
        layer1_error = derivate_tanh_act(layer1_out) * np.dot(layer2_error,weight1[:,1:])       
        weight1 = weight1 + learning_rate * np.dot(layer2_error.T, layer2_in)
        weight0 = weight0 + learning_rate * np.dot(layer1_error.T, layer1_in)   
    return (loss, weight0, weight1)
   

# Input    
x = np.array([[0,0], [0,1],[1,0],[1,1]])
y = np.array([[1,-1,-1,1]]).T
    
# Iterations
epochs = 50000
    
# Learning rate
learning_rate = 0.1
    
#Weights
weight0 = np.random.random((2,3))
weight1 = np.random.random((1,3))
    
    
# Training
print("Training")
(loss, weight0, weight1) = back_propagate(x,y,weight0,weight1,learning_rate,epochs)
print(loss)

print("\n",weight0)
print("\n",weight1)
    
# Validation
print("Testing")
layer1_in, layer1_sum, layer1_out, layer2_in, layer2_sum, y_cap = forward_propagate(x,weight0,weight1)
    
print("Result:\n")
print("Actual:\n",y,"\nPredicted:\n",y_cap)
    
# Plotting
plot_original_input(x,y)
plt.title("XOR input data visualization")
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.tight_layout()
plt.show()

# Plotting loss 
plt.title("Loss vs Epcohs")
plt.plot(loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()



plot_decision_regions(x, y, predict, weight0, weight1)
plt.title("Decision Boundary for XOR classification")
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.tight_layout()
plt.show()   
