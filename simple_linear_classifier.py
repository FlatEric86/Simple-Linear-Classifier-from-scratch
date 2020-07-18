import matplotlib.pyplot as plt
import numpy as np
import random as rand
import pandas as pd



# Number of trainingdata 
N = 500


# We want to build a simple one-neuron-linear-classifier, wich is able to classify
# if a point p(x, y) belongs to class A or to class B, dependent by their coordinates.
# We define the classes as gaussian distributed pointclouds around two different
# base points.
# These base points just represent the average point of their respected class
base_point_a = (3, 2)
base_point_b = (6, 3.5)


# Here we make randomized points and store them into a Pandas DataFrame
# The both first columns represents the coordinate (X, Y) and the third one
# is the "label".
# Means the class of the point.
df = pd.DataFrame(columns=['X', 'Y', 'C'])
for _ in range(N//2):

    for base_point in [base_point_a, base_point_b]:
        x = rand.gauss(base_point[0], 0.68)    # second value is the standard deviation, which can interpreted as the mean wideness
        y = rand.gauss(base_point[1], 0.8)
        
        if base_point == base_point_a:
            c = 'a'    # class A
        else:
            c = 'b'    # class B
            
            
        data = {
            'X':x, 
            'Y':y,
            'C':c    
        }
        
        df = df.append(data, ignore_index=True)
        
    



# A class which represents our Neuron as Object
class Neuron():
    def __init__(self):   
        # the weights of neuron which initialized values of (1,1,1)
        # the concept ist, that the third value represents the bias of the neuron
        self.weights = np.array([1, 1, 1])

    
    # That method trains the neuron by alter the weights by "learning" from given 
    # training data.
    # We use simply the Delta Rule 
    def train(self, t_data, N_epoch, lr, epsilon):
             
        # We iterate over a range of epochs determined by the N_epoch parameter
        for epoch in range(N_epoch):
               
            err = 0
            
            # Here we iterate over the training data
            for t in t_data:
            
                x = t[0]   # the x value of the point coordinate
                y = t[1]   # the y value of the point coordinate
                
                m_out = self.model_out(np.array([x, y]))  # the model output with actual weights      
                                                    
                err += (m_out - t[-1])**2                 # the absolut model error
            
                # alter weights by using delta rule
                self.weights = self.weights - lr*(m_out - t[-1])
            
            # Here we define, that we stop the iteration process if the model error is smaller
            # or equal to the handled parameter epsilon, if epsilon was defined as a number
            # That means, we can leave the iteration process if the condition was fullfilled
            
            if epsilon != None:
                if err < epsilon:
                    print('Number of iterations :', epoch)
                    break
            
             
    # That method represents the kernel method of the neuron
    # It returns the output of the model by given input data and
    # the weights.
    def model_out(self, X):
        X = np.append(X, 1)
        Y = np.dot(self.weights, X)
        
        if Y >= 1:
            return 1
        if Y < 1:
            return 0
        


# A Neuron object
neuron = Neuron()


# We transform the trainingdata to numpy arrays stored into a list
# to be able to use linear algebraic functions like dot product
# and transform the class labels to numerical values.
t_data_ = []
for x, y, c in zip(df['X'], df['Y'], df['C']):

    if c == 'a':
        Y = 0
    else:
        Y = 1
        
    t_data_.append(np.array([x, y, Y]))



# We train the Neuron object with the prepared (transformed) training data set
neuron.train(t_data_, 100, 0.01, None)


# a simple map to map our label value to a color value for plotting purposes
COLOR = {
    'a':'green',
    'b':'red'
}


# Here we plot our training data as scatter plot

flag = 0
for x, y, c in zip(df['X'], df['Y'], df['C']):

    
    plt.scatter(x, y, color=COLOR[c], alpha=0.3)
    
    flag = 1
    
    
    
# For validation we make new randomized data points and use 
# the Neuron to classify them    

# We use the same base parameter that we used for make the training data
base_point_a = (3, 2)
base_point_b = (6, 3.5)    
 
 
for n in range(30):
    for base_point in [base_point_a, base_point_b]:
    
        x = rand.gauss(base_point[0], 0.8)
        y = rand.gauss(base_point[1], 0.8)
        
        
        out = neuron.model_out(np.array([x, y]))
        
        if out == 0:
            color='green'
        else:
            color='red'
        plt.scatter(x, y, marker='+', color=color)  # we use other marker style
    
    
plt.xlabel('X')
plt.ylabel('Y')    
    
plt.show()

















