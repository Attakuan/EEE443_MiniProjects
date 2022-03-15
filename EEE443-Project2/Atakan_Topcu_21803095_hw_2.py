import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import h5py
import seaborn as sn
import sys

question = sys.argv[1]

def Atakan_Topcu_21803095_hw_2(question):
    if question == '1' :
        print("Question", question)
        ##question 1 code goes here
        filename = "assign2_data1.h5"
        file = h5py.File(filename, "r") #Load Data
        # List all groups
        print("Keys: %s" % list(file.keys()))
        testims = list(file.keys())[0]
        testlbls = list(file.keys())[1]
        trainims = list(file.keys())[2]
        trainlbls = list(file.keys())[3]

        # Get the data
        test_images = np.array(file[testims]).T
        test_labels = np.array(file[testlbls]).T
        train_images = np.array(file[trainims]).T
        train_labels = np.array(file[trainlbls]).T

        train_size = train_labels.shape[0]
        print("Number of Train Samples:",train_size) #Shows the number of train samples

        print("Train Image Size & Train Label Size:", train_images.shape,train_labels.shape)


        inputSize = train_images.shape[1]

        train_img_flat = train_images.reshape(inputSize**2,train_images.shape[2])
        test_img_flat = test_images.reshape(inputSize**2,test_images.shape[2])
        print("Train Image after reshaping:",train_img_flat.shape)
        print("Test Image after reshaping:",test_img_flat.shape)
        train_labels[train_labels == 0] = -1
        test_labels[test_labels == 0] = -1
        
        neuralNet = NeuralNetwork()
        neuralNet.addLayer(Layer(inputSize**2, 10, 0, 0.03, 1))
        neuralNet.addLayer(Layer(10, 1, 0, 0.03, 1))

        mses, mces, mseTs, mceTs = neuralNet.TrainNetwork(0.25, 57, train_img_flat/255,train_labels, test_img_flat/255, test_labels,400)
        print("Test Accuracy:", str(np.sum(neuralNet.Predict(test_img_flat/255) == test_labels)/len(test_labels)*100) + "%")


        fig, axs = plt.subplots(2, 2)

        axs[0, 0].plot(mses)
        axs[0, 0].set_title('MSE Over Training')
        axs[0, 0].set(ylabel='MSE')

        axs[0, 1].plot(mces)
        axs[0, 1].set_title('MCE Over Training')
        axs[0, 1].set(ylabel='MCE')

        axs[1, 0].plot(mseTs)
        axs[1, 0].set_title('MSE Over Test')
        axs[1, 0].set(xlabel='Epoch', ylabel='MSE')

        axs[1, 1].plot(mceTs)
        axs[1, 1].set_title('MCE Over Test')
        axs[1, 1].set(xlabel='Epoch', ylabel='MCE')
        fig.tight_layout(pad=1.0)
        plt.show()

        neuralNetHigh = NeuralNetwork()
        neuralNetHigh.addLayer(Layer(inputSize**2, 40, 0, 0.03, 1))
        neuralNetHigh.addLayer(Layer(40,1, 0, 0.03, 1))

        neuralNetLow = NeuralNetwork()
        neuralNetLow.addLayer(Layer(inputSize**2, 3, 0, 0.03, 1))
        neuralNetLow.addLayer(Layer(3, 1, 0, 0.03, 1))

        msesH, mcesH, mseTsH, mceTsH = neuralNetHigh.TrainNetwork(0.25, 57, train_img_flat/255, train_labels, test_img_flat/255, test_labels,400)
        msesL, mcesL, mseTsL, mceTsL = neuralNetLow.TrainNetwork(0.25, 57, train_img_flat/255, train_labels, test_img_flat/255, test_labels,400)

        plt.plot(mses)
        plt.plot(msesH)
        plt.plot(msesL)
        plt.title('MSE Over Training')
        plt.legend(['Orginal', 'High', 'Low'])
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.show()

        plt.plot(mces)
        plt.plot(mcesH)
        plt.plot(mcesL)
        plt.title('MCE Over Training')
        plt.legend(['Orginal', 'High', 'Low'])
        plt.xlabel('Epoch')
        plt.ylabel('MCE')
        plt.show()

        plt.plot(mseTs)
        plt.plot(mseTsH)
        plt.plot(mseTsL)
        plt.title('MSE Over Test')
        plt.legend(['Orginal', 'High', 'Low'])
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.show()

        plt.plot(mceTs)
        plt.plot(mceTsH)
        plt.plot(mceTsL)
        plt.title('MCE Over Test')
        plt.legend(['Orginal', 'High', 'Low'])
        plt.xlabel('Epoch')
        plt.ylabel('MCE')
        plt.show()


        neuralNetTwoHidden = NeuralNetwork()
        neuralNetTwoHidden.addLayer(Layer(inputSize**2, 70, 0, 0.03, 1))
        neuralNetTwoHidden.addLayer(Layer(70,20, 0, 0.03, 1))
        neuralNetTwoHidden.addLayer(Layer(20,1, 0, 0.03, 1))


        mses2, mces2, mseTs2, mceTs2 = neuralNetTwoHidden.TrainNetwork(0.3, 57, train_img_flat/255,train_labels, test_img_flat/255, test_labels,220)



        print("Test Accuracy:", str(np.sum(neuralNetTwoHidden.Predict(test_img_flat/255) == test_labels)/len(test_labels)*100) + "%")

        plt.plot(mses2)
        plt.title('MSE Over Training')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.show()

        plt.plot(mces2)
        plt.title('MCE Over Training')
        plt.xlabel('Epoch')
        plt.ylabel('MCE')
        plt.show()

        plt.plot(mseTs2)
        plt.title('MSE Over Test')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.show()

        plt.plot(mceTs2)
        plt.title('MCE Over Test')
        plt.xlabel('Epoch')
        plt.ylabel('MCE')
        plt.show()

        
        neuralNetTwoHiddenM = NeuralNetworkWithMomentum()
        neuralNetTwoHiddenM.addLayer(LayerWithMomentum(inputSize**2, 70, 0, 0.03, 1))
        neuralNetTwoHiddenM.addLayer(LayerWithMomentum(70,20, 0, 0.03, 1))
        neuralNetTwoHiddenM.addLayer(LayerWithMomentum(20,1, 0, 0.03, 1))
        MomentCoef=0.11
        mses2M, mces2M, mseTs2M, mceTs2M = neuralNetTwoHiddenM.TrainNetwork(0.3, 57,train_img_flat/255, train_labels, test_img_flat/255, test_labels,220,MomentCoef)


        print("Test Accuracy:", str(np.sum(neuralNetTwoHiddenM.Predict(test_img_flat/255) == test_labels)/len(test_labels)*100) + "%")
        

        plt.plot(mses2)
        plt.plot(mses2M)
        plt.legend(['wo/Momentum','w/Momentum'])
        plt.title('MSE Over Training')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.show()

        plt.plot(mces2)
        plt.plot(mces2M)
        plt.legend(['wo/Momentum','w/Momentum'])
        plt.title('MCE Over Training')
        plt.xlabel('Epoch')
        plt.ylabel('MCE')
        plt.show()

        plt.plot(mseTs2)
        plt.plot(mseTs2M)
        plt.legend(['wo/Momentum','w/Momentum'])
        plt.title('MSE Over Test')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.show()

        plt.plot(mceTs2)
        plt.plot(mceTs2M)
        plt.legend(['wo/Momentum','w/Momentum'])
        plt.title('MCE Over Test')
        plt.xlabel('Epoch')
        plt.ylabel('MCE')
        plt.show()

    elif question == '2' :
        print("Question", question)
        ##question 2 code goes here
        filename = "assign2_data2.h5"
        file = h5py.File(filename, "r") #Load Data

        # List all groups
        print("Keys: %s" % list(file.keys()))
        testd = list(file.keys())[0]
        testx = list(file.keys())[1]
        traind = list(file.keys())[2]
        trainx = list(file.keys())[3]
        vald = list(file.keys())[4]
        valx = list(file.keys())[5]
        words = list(file.keys())[6]

        # Get the data
        test_labels = np.array(file[testd])
        test_data = np.array(file[testx])
        train_labels = np.array(file[traind])
        train_data = np.array(file[trainx])
        val_labels = np.array(file[vald])
        val_data = np.array(file[valx])
        words = np.array(file[words])
        
        print("Train Data Size & Train Label Size:", train_data.shape,train_labels.shape)
        print("Test Data Size & Test Label Size:", test_data.shape,test_labels.shape)
        print("Validation Data Size & Validation Label Size:", val_data.shape,val_labels.shape)




        P = 256
        D = 32
        dictSize = 250

        nn = NeuralNetworkNLP()
        nn.addLayer(LayerNLP(D, dictSize,0,0.25, 'embedding'))
        nn.addLayer(LayerNLP(3*D, P, 0,0.25, 'sigmoid'))
        nn.addLayer(LayerNLP(P, dictSize, 0,0.25,'softmax'))

        learnRate = 0.15
        momCoeff = 0.70
        batchSize = 200
        epoch = 50


        val_data_One = SetupData(val_data, dictSize)
        val_labels_One = SetupLabel(val_labels, dictSize)

        errors = nn.TrainNetwork(learnRate,batchSize,train_data,train_labels,val_data_One,val_labels_One,epoch,momCoeff,dictSize)


        P2 = 128
        D2 = 16
        dictSize = 250

        nn2 = NeuralNetworkNLP()
        nn2.addLayer(LayerNLP(D2, dictSize,0,0.25, 'embedding'))
        nn2.addLayer(LayerNLP(3*D2, P2, 0,0.25, 'sigmoid'))
        nn2.addLayer(LayerNLP(P2, dictSize, 0,0.25,'softmax'))

        learnRate = 0.15
        momCoeff = 0.70
        batchSize = 200
        epoch = 50


        val_data_One = SetupData(val_data, dictSize)
        val_labels_One = SetupLabel(val_labels, dictSize)

        errors2 = nn2.TrainNetwork(learnRate,batchSize,train_data,train_labels,val_data_One,val_labels_One,epoch,momCoeff,dictSize)


        P3 = 64
        D3 = 8
        dictSize = 250

        nn3 = NeuralNetworkNLP()
        nn3.addLayer(LayerNLP(D3, dictSize,0,0.25, 'embedding'))
        nn3.addLayer(LayerNLP(3*D3, P3, 0,0.25, 'sigmoid'))
        nn3.addLayer(LayerNLP(P3, dictSize, 0,0.25,'softmax'))

        learnRate = 0.15
        momCoeff = 0.70
        batchSize = 200
        epoch = 50


        val_data_One = SetupData(val_data, dictSize)
        val_labels_One = SetupLabel(val_labels, dictSize)

        errors3 = nn3.TrainNetwork(learnRate,batchSize,train_data,train_labels,val_data_One,val_labels_One,epoch,momCoeff,dictSize)


        plt.plot(errors)
        plt.plot(errors2)
        plt.plot(errors3)
        plt.title('Cross-Entropy Error over Validation Set')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Error')
        plt.legend(['(32,256)', '(16,128)', '(8,64)'])
        plt.show()


        random_indexes = np.random.permutation(len(test_data))[0:5]

        test_samples = test_data[random_indexes,:]
        test_outputs = test_labels[random_indexes]

        test_samples_One = SetupData(test_samples, 250)

        top10 = nn.PredictTopK(test_samples_One, 10)

        for i in range(5):
            print('Sample ' + str(i+1)+ ": " + str(words[test_samples[i,0]-1].decode("utf-8"))+' ' +           str(words[test_samples[i,1]-1].decode("utf-8"))+' '            + str(words[test_samples[i,2]-1].decode("utf-8")))
            
            print('The Top 10 predictions: ')
            for j in range(10):
                top = ("["+str(j+1)+". "+ str(words[top10[j,i]-1].decode("utf-8")))+ "]"
                print(top)







class Layer:   
    def __init__(self,inputDim,numNeurons,mean,std,beta):
        self.inputDim = inputDim
        self.numNeurons = numNeurons
        self.beta = beta
        self.weights = np.random.normal(mean,std, inputDim*numNeurons).reshape(numNeurons, inputDim)
        self.biases = np.random.normal(mean,std, numNeurons).reshape(numNeurons,1)
        self.weightsAll = np.concatenate((self.weights, self.biases), axis=1)
        self.lastActiv=None
        self.lyrDelta=None
        self.lyrError=None
    def activation(self, x):
        #applying the hyperbolic tangent activation
        x=np.array(x)
        numSamples = x.shape[1]
        tempInp = np.r_[x, [np.ones(numSamples)*-1]]
        self.lastActiv = np.tanh(self.beta*np.matmul(self.weightsAll, tempInp))  
        return self.lastActiv

    def activation_derivative(self, x):
        #computing derivative 
        return self.beta*(1-(x**2))




class NeuralNetwork:
    def __init__(self):
        self.layers=[]
        
    def addLayer(self,layer):
        self.layers.append(layer)
    
    def FowardProp(self,training_inputs):
        #Foward Propagation
        IN=training_inputs    
        for layer in self.layers:
            IN=layer.activation(IN)
        return IN

    def BackProp(self,l_rate,batch_size,training_inputs,training_labels):   
        #Back Propagation
        foward_out = self.FowardProp(training_inputs)
        for i in reversed(range(len(self.layers))):
            #Output layer
            lyr=self.layers[i]
            if lyr == self.layers[-1]:
                lyr.lyrError=training_labels-foward_out
                derivative=lyr.activation_derivative(lyr.lastActiv)                
                lyr.lyrDelta=derivative*lyr.lyrError
            #Other layers
            else:
                nextLyr=self.layers[i+1]
                lyr.lyrError=np.matmul(nextLyr.weightsAll[:,0:nextLyr.weightsAll.shape[1]-1].T, nextLyr.lyrDelta)
                derivative=lyr.activation_derivative(lyr.lastActiv)               
                lyr.lyrDelta=derivative*lyr.lyrError
                
        #UPDATE THE WEIGHT MATRIX
        for i in (range(len(self.layers))):
            lyr=self.layers[i]
            if i==0:
                numSamples = training_inputs.shape[1]
                tempInp = np.r_[training_inputs, [np.ones(numSamples)*-1]]
            else:
                prevLyr=self.layers[i-1]
                numSamples=prevLyr.lastActiv.shape[1],
                tempInp = np.r_[prevLyr.lastActiv, [np.ones(numSamples)*-1]]
            
            
            lyr.weightsAll=lyr.weightsAll+l_rate*np.matmul(lyr.lyrDelta, tempInp.T)/batch_size
        
    def TrainNetwork(self,l_rate,batch_size,training_inputs,training_labels, test_inputs, test_labels, epochNum):
        mseList = []
        mceList = []
        mseTestList = []
        mceTestList = []           
        for epoch in range(epochNum):
            print("Epoch:",epoch)
            indexing=np.random.permutation(training_inputs.shape[1])
            #Randomly mixing the samples
            training_inputs=training_inputs[:,indexing]
            training_labels=training_labels[indexing]
            numBatches = int(np.floor(training_inputs.shape[1]/batch_size)) 
            for j in range(numBatches):
                self.BackProp(l_rate,batch_size,training_inputs[:,j*numBatches:numBatches*(j+1)],training_labels[j*numBatches:numBatches*(j+1)])         
            
            mse = np.mean((training_labels - self.FowardProp(training_inputs))**2)
            mseList.append(mse)
            mce = np.sum(self.Predict(training_inputs) == training_labels)/len(training_labels)*100
            mceList.append(mce)
            mseT = np.mean((test_labels - self.FowardProp(test_inputs))**2)
            mseTestList.append(mseT)
            mceT = np.sum(self.Predict(test_inputs) == test_labels)/len(test_labels)*100
            mceTestList.append(mceT)
        return mseList, mceList, mseTestList, mceTestList
                
    def Predict(self,inputIMG):
        out = self.FowardProp(inputIMG)
        out[out>=0] = 1
        out[out<0] = -1
        return out 
    

class LayerWithMomentum:
    def __init__(self,inputDim,numNeurons,mean,std,beta):
        self.inputDim = inputDim
        self.numNeurons = numNeurons
        self.beta = beta
        self.weights = np.random.normal(mean,std, inputDim*numNeurons).reshape(numNeurons, inputDim)
        self.biases = np.random.normal(mean,std, numNeurons).reshape(numNeurons,1)
        self.weightsAll = np.concatenate((self.weights, self.biases), axis=1)
        self.lastActiv=None
        self.lyrDelta=None
        self.lyrError=None
        self.prevUpdate = 0
    def activation(self, x):
        #applying the hyperbolic tangent activation
        x=np.array(x)
        numSamples = x.shape[1]
        tempInp = np.r_[x, [np.ones(numSamples)*-1]]
        self.lastActiv = np.tanh(self.beta*np.matmul(self.weightsAll, tempInp))  
        return self.lastActiv

    def activation_derivative(self, x):
        #computing derivative 
        return self.beta*(1-(x**2)) 
    
    

class NeuralNetworkWithMomentum:
    def __init__(self):
        self.layers=[]
        
    def addLayer(self,layer):
        self.layers.append(layer)
    
    def FowardProp(self,training_inputs):
        #Foward Propagation
        IN=training_inputs    
        for layer in self.layers:
            IN=layer.activation(IN)
        return IN

    def BackProp(self,l_rate,batch_size,training_inputs,training_labels,momentCoef):   
        #Back Propagation
        foward_out = self.FowardProp(training_inputs)
        for i in reversed(range(len(self.layers))):
            #Output layer
            lyr=self.layers[i]
            if lyr == self.layers[-1]:
                lyr.lyrError=training_labels-foward_out
                derivative=lyr.activation_derivative(lyr.lastActiv)                
                lyr.lyrDelta=derivative*lyr.lyrError
            #Other layers
            else:
                nextLyr=self.layers[i+1]
                lyr.lyrError=np.matmul(nextLyr.weightsAll[:,0:nextLyr.weightsAll.shape[1]-1].T, nextLyr.lyrDelta)
                derivative=lyr.activation_derivative(lyr.lastActiv)               
                lyr.lyrDelta=derivative*lyr.lyrError
                
        #UPDATE THE WEIGHT MATRIX
        for i in (range(len(self.layers))):
            lyr=self.layers[i]
            if i==0:
                numSamples = training_inputs.shape[1]
                tempInp = np.r_[training_inputs, [np.ones(numSamples)*-1]]
            else:
                prevLyr=self.layers[i-1]
                numSamples=prevLyr.lastActiv.shape[1],
                tempInp = np.r_[prevLyr.lastActiv, [np.ones(numSamples)*-1]]
            
            update =  l_rate*np.matmul(lyr.lyrDelta, tempInp.T)/batch_size
            lyr.weightsAll+= update + (momentCoef*lyr.prevUpdate)
            lyr.prevUpdate = update
        
    def TrainNetwork(self,l_rate,batch_size,training_inputs,training_labels, test_inputs, test_labels, epochNum,momentCoef):
        mseList = []
        mceList = []
        mseTestList = []
        mceTestList = []           
        for epoch in range(epochNum):
            print("Epoch:",epoch)
            indexing=np.random.permutation(training_inputs.shape[1])
            #Randomly mixing the samples
            training_inputs=training_inputs[:,indexing]
            training_labels=training_labels[indexing]
            numBatches = int(np.floor(training_inputs.shape[1]/batch_size)) 
            for j in range(numBatches):
                self.BackProp(l_rate,batch_size,training_inputs[:,j*numBatches:numBatches*(j+1)],training_labels[j*numBatches:numBatches*(j+1)],momentCoef)         
            
            mse = np.mean((training_labels - self.FowardProp(training_inputs))**2)
            mseList.append(mse)
            mce = np.sum(self.Predict(training_inputs) == training_labels)/len(training_labels)*100
            mceList.append(mce)
            mseT = np.mean((test_labels - self.FowardProp(test_inputs))**2)
            mseTestList.append(mseT)
            mceT = np.sum(self.Predict(test_inputs) == test_labels)/len(test_labels)*100
            mceTestList.append(mceT)
        return mseList, mceList, mseTestList, mceTestList
                
    def Predict(self,inputIMG):
        out = self.FowardProp(inputIMG)
        out[out>=0] = 1
        out[out<0] = -1
        return out 


#One-hot encoding
def SetupLabel(y, dictSize):
    out = np.zeros((y.shape[0], dictSize))
    for i in range(y.shape[0]):
        out1 = np.zeros(dictSize)
        out1[y[i]-1] = 1               
        out[i,:] = out1
    return out

def SetupData(x, dictSize):
    out = np.zeros((x.shape[0], x.shape[1], dictSize))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out1 = np.zeros(dictSize)
            out1[x[i,j]-1] = 1             
            out[i,j,:] = out1
    return out




class LayerNLP: #Modified Version of Class in Q1
    def __init__(self,inputDim,numNeurons,mean,std, activation):
        self.inputDim = inputDim
        self.numNeurons = numNeurons
        self.activation = activation
        if self.activation == 'sigmoid' or self.activation == 'softmax':
            self.weights = np.random.normal(mean,std, inputDim*numNeurons).reshape(numNeurons, inputDim)
            self.biases = np.random.normal(mean,std, numNeurons).reshape(numNeurons,1)
            self.weightsAll = np.concatenate((self.weights, self.biases), axis=1)
        else:
            self.dictSize = numNeurons
            self.D = inputDim
            self.weights = np.random.normal(mean, std, self.dictSize*self.D).reshape((self.dictSize,self.D))            
        
        self.lastActiv=None

        self.lyrDelta=None
        self.lyrError=None
        self.prevUpdate = 0
        
    def activationFunction(self, x):
        if(self.activation == 'sigmoid'):
            exp_x = np.exp(2*x)
            return exp_x/(1+exp_x)
        elif(self.activation == 'softmax'):
            exp_x = np.exp(x - np.max(x))
            return exp_x/np.sum(exp_x, axis=0)
        else:
            return x
        
    def activationNeuron(self,x):     
        if self.activation == 'sigmoid' or self.activation == 'softmax':                
            numSamples = x.shape[1]
            tempInp = np.r_[x, [np.ones(numSamples)*-1]]    
            self.lastActiv = self.activationFunction(np.matmul(self.weightsAll, tempInp))
            
        else:
            EmbedOut = np.zeros((x.shape[0],x.shape[1], self.D))
            for m in range(EmbedOut.shape[0]): #For each sample
                EmbedOut[m,:,:] = self.activationFunction(np.matmul(x[m,:,:], self.weights))
            EmbedOut = EmbedOut.reshape((EmbedOut.shape[0], EmbedOut.shape[1] * EmbedOut.shape[2]))
            self.lastActiv = EmbedOut.T #For adjusting to other layer's input parameters. 
                                        #Otherwise, it will yield error.
        return self.lastActiv        

    def activation_derivative(self, x):
        if(self.activation == 'sigmoid'):
            return (x*(1-x))
        elif(self.activation == 'softmax'):
            return x*(1-x)
        else:
            return np.ones(x.shape)
    




class NeuralNetworkNLP: #Modified Version of Class in Q1
 def __init__(self):
     self.layers=[]
     
 def addLayer(self,layer):
     self.layers.append(layer)
 
 def FowardProp(self,training_inputs):
     #Foward Propagation
     IN=training_inputs    
     for layer in self.layers:
         IN=layer.activationNeuron(IN)
     return IN

 def BackProp(self,l_rate,batch_size,training_inputs,training_labels,momentCoef):   
     foward_out = self.FowardProp(training_inputs)
     for i in reversed(range(len(self.layers))):
         lyr = self.layers[i]
         #outputLayer
         if(lyr == self.layers[-1]):
             lyr.lyrDelta=training_labels.T-foward_out     
         else:
             nextLyr = self.layers[i+1]
             lyr.lyrError = np.matmul(nextLyr.weights.T, nextLyr.lyrDelta)
             derivative=lyr.activation_derivative(lyr.lastActiv)               
             lyr.lyrDelta=derivative*lyr.lyrError
             
     
     #update weights
     for i in range(len(self.layers)):
         lyr = self.layers[i]
         if(i == 0):
             tempInp = training_inputs
         else:
             numSamples = self.layers[i - 1].lastActiv.shape[1]
             tempInp = np.r_[self.layers[i - 1].lastActiv, [np.ones(numSamples)*-1]]
         if(lyr.activation == 'sigmoid' or lyr.activation == 'softmax'):
             update =  l_rate*np.matmul(lyr.lyrDelta, tempInp.T)/batch_size
             lyr.weightsAll+= update + (momentCoef*lyr.prevUpdate)
         else:          
             deltaEmbed = lyr.lyrDelta.reshape((3,batch_size,lyr.D))
             tempInp = np.transpose(tempInp, (1,0,2)) #Rotating the input 
             update = np.zeros((tempInp.shape[2], deltaEmbed.shape[2]))
             for i in range(deltaEmbed.shape[0]):
                 update += l_rate * np.matmul(tempInp[i,:,:].T, deltaEmbed[i,:,:])
             update = update/batch_size
             lyr.weights += update + (momentCoef*lyr.prevUpdate)
         lyr.prevUpdate = update
         
 def TrainNetwork(self,l_rate,batch_size,training_inputs,training_labels, test_inputs, test_labels, epochNum,momentCoef,dictSize):
     crossList = []    
     for epoch in range(epochNum):
         print("Epoch:",epoch)
         indexing=np.random.permutation(len(training_inputs))
         #Randomly mixing the samples
         training_inputs=training_inputs[indexing,:]
         training_labels=training_labels[indexing]
         numBatches = int(np.floor(len(training_inputs)/batch_size)) 
         for j in range(numBatches):
             train_data_One = SetupData(training_inputs[j*batch_size:batch_size*(j+1),:], dictSize)
             train_labels_One = SetupLabel(training_labels[j*batch_size:batch_size*(j+1)], dictSize)
             self.BackProp(l_rate,batch_size,train_data_One,train_labels_One,momentCoef)         
         
         valOutput = self.FowardProp(test_inputs)
         crossErr = - np.sum(np.log(valOutput) * test_labels.T)/valOutput.shape[1]
         print('Cross-Entropy Error ', crossErr)
         crossList.append(crossErr)

     return crossList
             
 
 def PredictTopK(self, inputIMG, k):
     out = self.FowardProp(inputIMG)
     return np.argsort(out, axis=0)[:,0:k]



Atakan_Topcu_21803095_hw_2(question)
