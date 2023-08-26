import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


data_bc = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",header=None)
data_iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

dic_bc_class = {
    'B' : 0,
    'M' : 1
}

dic_iris_class = {
    'Iris-setosa' : 0,
    'Iris-versicolor' : 1,
    'Iris-virginica': 2
}

dict_live1 = { 
    0 : 'Iris-setosa',
    1 : 'Iris-versicolor',
    2 : 'Iris-virginica'
}

dict_live = { 
    0 : 'B',
    1 : 'M',
}

data_bc[1] = data_bc[1].apply(lambda x : dic_bc_class[x])
data_iris['Bclass'] = data_iris['class'].apply(lambda x : dic_iris_class[x])

features_bc = data_bc[[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 ]].to_numpy()
labels_bc = data_bc[1].to_numpy()
features_iris = data_iris[['sepal_length','sepal_width','petal_length','petal_width']].to_numpy()
labels_iris = data_iris['Bclass'].to_numpy()

#Splitting dataset into training and testing set
bc_X_train, bc_X_test, bc_Y_train, bc_Y_test = train_test_split(features_bc, labels_bc, test_size=0.20)
iris_X_train, iris_X_test, iris_Y_train, iris_Y_test = train_test_split(features_iris, labels_iris, test_size=0.20)

print("Breast Cancer Data set")
print('Training sets:',bc_X_train.size)
print('Test sets:',bc_X_test.size)
print('Training sets:',bc_Y_train.size)
print('Test sets:',bc_Y_test.size)

print("Iris Data set")
print('Training sets:',iris_X_train.size)
print('Test sets:',iris_X_test.size)
print('Training sets:',iris_Y_train.size)
print('Test sets:',iris_Y_test.size)


#Multilayer class
class ANN:
    np.random.seed(10)
    def __init__(self) :
        #Layers Information
        self.HiddenLayer = []
        #weight list
        self.w = []
        #bias list
        self.b = []
        #Activation Function
        self.phi = []
        #Cost function
        self.mu = []
        #Learning rate
        self.eta = 1 
        #momentum
        self.momentum = 0.5
        #batch size
        self.batch = 10
        #Epoch
        self.epoch= 600
    
    #Add layer to network
    def add(self, lay = (4, 'tanh') ):
        self.HiddenLayer.append(lay)
    
    
    #FeedForward
    @staticmethod
    def FeedForward(w, b, phi, x):
        return phi(np.dot(w, x) + b)
        
   
   #BackPropagation
    def BackPropagation(self, x, z, Y, w, b, phi):
        self.delta = []
        
        # We initialize w and b that are used only inside the backpropagation algorithm once called        
        self.W = []
        self.B = []
        
        # We start computing the LAST error, the one for the OutPut Layer 
        self.delta.append(  (z[len(z)-1] - Y) * phi[len(z)-1](z[len(z)-1], der=True) )
        
        
        # We thus compute from next-to-last to first
        for i in range(0, len(z)-1):
            self.delta.append( self.momentum*np.dot( self.delta[i], w[len(z)- 1 - i] ) * phi[len(z)- 2 - i](z[len(z)- 2 - i], der=True) )
        
        # We have the error array ordered from last to first; we flip it to order it from first to last
        self.delta = np.flip(self.delta, 0)  
        
        # Now we define the delta as the error divided by the number of training samples
        self.delta = self.delta/self.X.shape[0] 
        
        '''GRADIENT DESCENT'''
        # We start from the first layer that is special, since it is connected to the Input Layer
        self.W.append( w[0] - self.eta * np.kron(self.delta[0], x).reshape( len(z[0]), x.shape[0] ) )
        self.B.append( b[0] - self.eta * self.delta[0] )
        
        # We now descend for all the other Hidden Layers + OutPut Layer
        for i in range(1, len(z)):
            self.W.append( w[i] - self.eta * np.kron(self.delta[i], z[i-1]).reshape(len(z[i]), len(z[i-1])) )
            self.B.append( b[i] - self.eta * self.delta[i] )
        
        # We return the descended parameters w, b
        return np.array(self.W), np.array(self.B)
    
    

    #Fit method for training: it calls FeedForward and Backpropagation methods
    def Fit(self, X_train, Y_train):            
        print('Start fitting...')
        
        #Input layer
        self.X = X_train
        self.Y = Y_train
        
        #We now initialize the Network by retrieving the Hidden Layers and concatenating them 
        print('Model recap: \n')
        print('You are fitting an ANN with the following amount of layers: ', len(self.HiddenLayer))
        
        for i in range(0, len(self.HiddenLayer)) :
            print('Layer ', i+1)
            print('Number of neurons: ', self.HiddenLayer[i][0])
            if i==0:
                # We now try to use the He et al. Initialization from ArXiv:1502.01852
                self.w.append( np.random.randn(self.HiddenLayer[i][0] , self.X.shape[1])/np.sqrt(2/self.X.shape[1]) )
                self.b.append( np.random.randn(self.HiddenLayer[i][0])/np.sqrt(2/self.X.shape[1]))
                # Old initialization
                #self.w.append(2 * np.random.rand(self.HiddenLayer[i][0] , self.X.shape[1]) - 0.5)
                #self.b.append(np.random.rand(self.HiddenLayer[i][0]))
                
                # Initialize the Activation function
                for act in Activation_function.list_act():
                    if self.HiddenLayer[i][1] == act :
                        self.phi.append(Activation_function.get_act(act))
                        print('\tActivation: ', act)

            else :
                # We now try to use the He et al. Initialization from ArXiv:1502.01852
                self.w.append( np.random.randn(self.HiddenLayer[i][0] , self.HiddenLayer[i-1][0] )/np.sqrt(2/self.HiddenLayer[i-1][0]))
                self.b.append( np.random.randn(self.HiddenLayer[i][0])/np.sqrt(2/self.HiddenLayer[i-1][0]))
                # Old initialization
                #self.w.append(2*np.random.rand(self.HiddenLayer[i][0] , self.HiddenLayer[i-1][0] ) - 0.5)
                #self.b.append(np.random.rand(self.HiddenLayer[i][0]))
                
                # Initialize the Activation function
                for act in Activation_function.list_act():
                    if self.HiddenLayer[i][1] == act :
                        self.phi.append(Activation_function.get_act(act))
                        print('\tActivation: ', act)
            
         
        # loop over the training set
        for I in range(self.epoch*0, self.X.shape[0]): 

            #Feedforward start 
            self.z = []
            
            # First layers
            self.z.append( self.FeedForward(self.w[0], self.b[0], self.phi[0], self.X[I]) ) 
            
            #Looping over layers
            for i in range(1, len(self.HiddenLayer)): 
                self.z.append( self.FeedForward(self.w[i] , self.b[i], self.phi[i], self.z[i-1] ) )
        
            
           
            #backpropagte      
            self.w, self.b  = self.BackPropagation(self.X[I], self.z, self.Y[I], self.w, self.b, self.phi)
            
          
            # Compute cost function 
            self.mu.append(
                (1/2) * np.dot(self.z[len(self.z)-1] - self.Y[I], self.z[len(self.z)-1] - self.Y[I]) 
            )
            
        print('Fit done. \n')
        

    
    #prediction method
    def predict(self, X_test):
        
        print('Starting predictions...')
        
        self.pred = []
        self.XX = X_test
        
        #loop over the training set
        for I in range(0, self.XX.shape[0]): 
            
            #Feedforward start
            self.z = []
            
            #First layer
            self.z.append(self.FeedForward(self.w[0] , self.b[0], self.phi[0], self.XX[I])) 
            
            # loop over the layers
            for i in range(1, len(self.HiddenLayer)) : 
                self.z.append( self.FeedForward(self.w[i] , self.b[i], self.phi[i], self.z[i-1]))
       
            # Append the prediction;
            # We now need a binary classifier; we do this apply an Heaviside Theta and we set to 0.5 the threshold
            # if y < 0.5 the output is zero, otherwise is zero
            # NB: self.z[-1]  is the last element of the self.z list
            self.pred.append( np.heaviside(  self.z[-1] - 0.5, 1)[0] ) 
        
        print('Predictions done. \n')

        return np.array(self.pred)
   
    #Accuracy
    def get_accuracy(self):
        return np.array(self.mu)
    
    # This is the averaged version
    def get_avg_accuracy(self):
        import math
        self.batch_loss = []
        for i in range(self.batch):
            self.loss_avg = 0
            # To set the batch in 10 element/batch we use math.ceil method
            # int(math.ceil((self.X.shape[0]-10) / 10.0))    - 1
            for m in range(0, (int(math.ceil((self.X.shape[0]-10) / 10.0))   )-1):
                #self.loss_avg += self.mu[60*i+m]/60
                self.loss_avg += self.mu[(int(math.ceil((self.X.shape[0]-10) / 10.0)) )*i + m]/(int(math.ceil((self.X.shape[0]-10) / 10.0)) )
            self.batch_loss.append(self.loss_avg)
        return np.array(self.batch_loss)
    
    
    #Method to set the learning rate
    def set_learning_rate(self, et=1):
        self.eta = et
    
    #Method to set the momentum rate
    def set_momentum(self, mt=0.5):
        self.momentum = mt

    #Method to set the batch size
    def set_batch(self, bat=10):
        self.batch = bat

    #Method to set the learning rate
    def set_epoch(self, epoch=600):
        self.epoch = epoch
        

# Define the sigmoid activator; we ask if we want the sigmoid or its derivative
def sigmoid_act(x, der=False):
    
    #derivative of the sigmoid
    if (der==True) : 
        f = 1/(1+ np.exp(- x))*(1-1/(1+ np.exp(- x)))
    else : # sigmoid
        f = 1/(1+ np.exp(- x))
    
    return f

# We may employ the Hyperbolic tabgent (Tanh)
def tanh_act(x, der=False):
    
    # the derivative of the Tanh
    if (der == True): 
        f = 1 - x**2
    else :
        f = np.tanh(x)
    
    return f

#layers   
class layers :
           
    def layer(p=4, activation = 'tanh'):
        return (p, activation)


#Activation functions class
class Activation_function(ANN):
    
    
    def __init__(self) :
        super().__init__()
        
    #sigmoid activation
    def sigmoid_act(x, der=False):
        if (der==True) : #derivative of the sigmoid
            f = 1/(1+ np.exp(- x))*(1-1/(1+ np.exp(- x)))
        else : # sigmoid
            f = 1/(1+ np.exp(- x))
        return f

    #tanh activation
    def tanh_act(x, der=False):
        if (der == True): # the derivative of the Tanh
            f = 1 - x**2
        else :
            f = np.tanh(x)
        return f
    
    def list_act():
        return ['sigmoid', 'tanh']
    
    def get_act(string = 'tanh'):
        if string == 'tanh':
            return tanh_act
        elif string == 'sigmoid':
            return sigmoid_act
        else :
            return sigmoid_act
        
#model = ANN()
        
#model.add(layers.layer(30, 'tanh'))
#model.add(layers.layer(15, 'tanh'))
#model.add(layers.layer(7, 'tanh'))
#for i in range(0,2):
        #neuron = input("Number of neuron of hidden " + i)
        #model.add(layers.layer(12, 'sigmoid'))
#model.add(layers.layer(2, 'tanh'))

#model.set_learning_rate(0.05)

#model.Fit(iris_X_train, iris_Y_train)
#acc_val = model.get_accuracy()
#acc_avg_val = model.get_avg_accuracy()

#model.Fit(bc_X_train, bc_Y_train)
#acc_val = model.get_accuracy()
#acc_avg_val = model.get_avg_accuracy()

#For Iris data set
#plt.figure(figsize=(10,6))
#plt.scatter(np.arange(1, iris_X_train.shape[0]+1), acc_val, alpha=0.3, s=4, label='mu')
#plt.title('Loss for each training data point', fontsize=20)
#plt.xlabel('Training data', fontsize=16)
#plt.ylabel('Loss', fontsize=16)
#plt.show()

#for breast cancer
#plt.figure(figsize=(10,6))
#plt.scatter(np.arange(1, bc_X_train.shape[0]+1), acc_val, alpha=0.3, s=4, label='mu')
#plt.title('Loss for each training data point', fontsize=20)
#plt.xlabel('Training data', fontsize=16)
#plt.ylabel('Loss', fontsize=16)
#plt.show()

#plt.figure(figsize=(10,6))
#plt.scatter(np.arange(1, len(acc_avg_val)+1), acc_avg_val, label='mu')
#plt.title('Averege Loss by epoch', fontsize=20)
#plt.xlabel('Training data', fontsize=16)
#plt.ylabel('Loss', fontsize=16)
#plt.show()

#predictions = model.predict(iris_X_test)
# Plot the confusion matrix
#cm = confusion_matrix(iris_Y_test, predictions)

#predictions = model.predict(bc_X_test)
# Plot the confusion matrix
#cm = confusion_matrix(bc_Y_test, predictions)

#df_cm = pd.DataFrame(cm, index = [dict_live1[i] for i in range(0,3)], columns = [dict_live1[i] for i in range(0,3)])
#plt.figure(figsize = (7,7))
#sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
#plt.xlabel("Predicted Class", fontsize=18)
#plt.ylabel("True Class", fontsize=18)
#plt.show()

#df_cm = pd.DataFrame(cm, index = [dict_live[i] for i in range(0,2)], columns = [dict_live[i] for i in range(0,2)])
#plt.figure(figsize = (7,7))
#sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
#plt.xlabel("Predicted Class", fontsize=18)
#plt.ylabel("True Class", fontsize=18)
#plt.show()

#ac = accuracy_score(iris_Y_test, predictions)
#print("For iris" +ac)

#ac = accuracy_score(bc_Y_test, predictions)
#print(ac)

if __name__== "__main__":
    model = ANN()

    Activation_name = input("Enter the activation function name: ")
    model.add(layers.layer(30, Activation_name))
    hidden = int(input('Enter the number of hidden layer: '))
    for i in range(hidden):
        neuron = int(input("Number of neuron of hidden{0}: " .format(i+1)))
        model.add(layers.layer(neuron, Activation_name))
    
    model.add(layers.layer(2, Activation_name))
    #learning rate
    learningrate = float(input('Enter a learning rate number: '))
    model.set_learning_rate(learningrate)
    #EPOCH
    epoch = int(input('Enter the number of Epoch: '))
    model.set_epoch(epoch)
    #batch
    batch = int(input('Enter the number of Batch: '))
    model.set_batch(batch)

    #momentum
    momentum = float(input('Enter the momentum rate: '))
    model.set_momentum(momentum)

    model.Fit(bc_X_train, bc_Y_train)
    acc_val = model.get_accuracy()
    acc_avg_val = model.get_avg_accuracy()

    plt.figure(figsize=(10,6))
    plt.scatter(np.arange(1, bc_X_train.shape[0]+1), acc_val, alpha=0.3, s=4, label='mu')
    plt.title('Loss for each training data point', fontsize=20)
    plt.xlabel('Training data', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.show()

    plt.figure(figsize=(10,6))
    plt.scatter(np.arange(1, len(acc_avg_val)+1), acc_avg_val, label='mu')
    plt.title('Averege Loss by epoch', fontsize=20)
    plt.xlabel('Training data', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.show()

    predictions = model.predict(bc_X_test)
    # Plot the confusion matrix
    cm = confusion_matrix(bc_Y_test, predictions)

    df_cm = pd.DataFrame(cm, index = [dict_live[i] for i in range(0,2)], columns = [dict_live[i] for i in range(0,2)])
    plt.figure(figsize = (7,7))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.xlabel("Predicted Class", fontsize=18)
    plt.ylabel("True Class", fontsize=18)
    plt.show()

#ac = accuracy_score(iris_Y_test, predictions)
#print("For iris" +ac)

    ac = accuracy_score(bc_Y_test, predictions)
    print(ac)
