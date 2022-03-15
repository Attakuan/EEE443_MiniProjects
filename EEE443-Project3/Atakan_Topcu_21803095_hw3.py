

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import h5py
import seaborn as sn
filename= "assign3_data1.h5"
filename_Q3 = "assign3_data3.h5"
import sys

question = sys.argv[1]

def Atakan_Topcu_21803095_hw3(question):
    if question == '1' :
        ##question 1 code goes here
        
        file = h5py.File(filename, "r") #Load Data
        # List all groups
        print("Keys: %s" % list(file.keys()))
        data = list(file.keys())[0]
        invXForm = list(file.keys())[1]
        xForm = list(file.keys())[2]

        # Get the data
        data = np.array(file[data])
        invXForm = np.array(file[invXForm])
        xForm = np.array(file[xForm])
        
        print('Data:', data.shape)
        PreprocessedData=0.2126*data[:,0,:,:]+0.7152*data[:,1,:,:]+0.0722*data[:,0,:,:]
        print('Preprocessed Data:', PreprocessedData.shape)
        meanIntensity = np.mean(PreprocessedData, axis=(1,2))
        NormalizedData = np.zeros(PreprocessedData.shape)
        #Mean pixel intensity subtraction
        for i in range(PreprocessedData.shape[0]):
                NormalizedData[i,:,:]=PreprocessedData[i,:,:]-meanIntensity[i]
        #Standart deviation
        std = np.std(NormalizedData)
        NormalizedData = np.minimum(NormalizedData, 3*std)
        NormalizedData = np.maximum(NormalizedData, -3*std)
        #[0.1 0.9] scaling
        minRange=0.1
        maxRange=0.9
        difference=maxRange-minRange
        #Adjusting the scale to the range [0.1 0.9]
        NormalizedData=minRange+((NormalizedData-np.min(NormalizedData))*difference)/(2*np.max(NormalizedData))
        print(np.min(NormalizedData))
        print(np.max(NormalizedData))

        #Plotting
        row = 10
        col = 20
        fig = plt.figure(figsize=(20, 15), dpi= 160)

        TotalSampleSize = NormalizedData.shape[0]
        randInd = np.random.permutation(TotalSampleSize)

        for i in range(row*col):
                ax = plt.subplot(row, col, i+1)
                plt.imshow(data[randInd[i],:,:,:].T)
                ax.axis('off')
        plt.show()

        
        fig=plt.figure(figsize=(20, 15), dpi= 160)        
        for j in range(row*col):
                ax = plt.subplot(row, col, j+1)
                plt.imshow(NormalizedData[randInd[j],:,:].T, cmap='gray')
                ax.axis('off')
        plt.show()       

        
        #Flatten images to represent each pixel as neuron.
        Pixel = NormalizedData.shape[1]
        SampleSize=NormalizedData.shape[0]
        Flat_Data = np.reshape(NormalizedData, (SampleSize,Pixel**2))
        print("Flattened Data Shape:", Flat_Data.shape)
        Linput=Loutput=Pixel**2
        Lhid = 64
        lmb = 5e-4
        beta = 0.06
        rho = 0.2
        params = (Linput, Lhid, lmb, beta, rho) 
        
        ae=Autoencoder()
        We=ae.WeightInitialization(Linput,Lhid,Loutput)
        We_update=ae.Train(We,Flat_Data,params,0.25,200,640)
        # # Part C
        row = 8
        col = 8
        We_final=We_update
        fig=plt.figure(figsize=(10, 10))
        for i in range(We_final[0].shape[1]):
            plt.subplot(row,col,i+1)
            plt.imshow(np.reshape(We_final[0][:,i],(Pixel,Pixel)), cmap='gray')
            plt.axis('off')
        plt.show()




        # # Part D

        # In[196]:


        #Different Hidden Neuron Numbers

        Pixel = NormalizedData.shape[1]
        SampleSize=NormalizedData.shape[0]
        Flat_Data = np.reshape(NormalizedData, (SampleSize,Pixel**2))

        Linput=Loutput=Pixel**2
        Lhid = 12 #Changed
        lmb = 5e-4
        beta = 0.06
        rho = 0.2
        params = (Linput, Lhid, lmb, beta, rho)

        ae_12=Autoencoder()
        We_12=ae_12.WeightInitialization(Linput,Lhid,Loutput)
        We_update_12=ae_12.Train(We_12,Flat_Data,params,0.25,200,640)


        # In[197]:


        row = 3
        col = 4
        We_final=We_update_12
        fig=plt.figure(figsize=(10, 10))
        for i in range(We_final[0].shape[1]):
            plt.subplot(row,col,i+1)
            plt.imshow(np.reshape(We_final[0][:,i],(Pixel,Pixel)), cmap='gray')
            plt.axis('off')
        plt.show()


        # In[198]:


        #Different Hidden Neuron Numbers

        Pixel = NormalizedData.shape[1]
        SampleSize=NormalizedData.shape[0]
        Flat_Data = np.reshape(NormalizedData, (SampleSize,Pixel**2))

        Linput=Loutput=Pixel**2
        Lhid = 48 #Changed
        lmb = 5e-4
        beta = 0.06
        rho = 0.2
        params = (Linput, Lhid, lmb, beta, rho)

        ae_48=Autoencoder()
        We_48=ae_48.WeightInitialization(Linput,Lhid,Loutput)
        We_update_48=ae_48.Train(We_48,Flat_Data,params,0.25,200,640)


        # In[199]:


        row = 6
        col = 8
        We_final=We_update_48
        fig=plt.figure(figsize=(10, 10))
        for i in range(We_final[0].shape[1]):
            plt.subplot(row,col,i+1)
            plt.imshow(np.reshape(We_final[0][:,i],(Pixel,Pixel)), cmap='gray')
            plt.axis('off')
        plt.show()


        # In[200]:


        #Different Hidden Neuron Numbers

        Pixel = NormalizedData.shape[1]
        SampleSize=NormalizedData.shape[0]
        Flat_Data = np.reshape(NormalizedData, (SampleSize,Pixel**2))

        Linput=Loutput=Pixel**2
        Lhid = 96 #Changed
        lmb = 5e-4
        beta = 0.06
        rho = 0.2
        params = (Linput, Lhid, lmb, beta, rho)

        ae_96=Autoencoder()
        We_96=ae_96.WeightInitialization(Linput,Lhid,Loutput)
        We_update_96=ae_96.Train(We_96,Flat_Data,params,0.25,200,640)


        # In[201]:


        row = 8
        col = 12
        We_final=We_update_96
        fig=plt.figure(figsize=(10, 10))
        for i in range(We_final[0].shape[1]):
            plt.subplot(row,col,i+1)
            plt.imshow(np.reshape(We_final[0][:,i],(Pixel,Pixel)), cmap='gray')
            plt.axis('off')
        plt.show()


        # In[11]:


        #Different Lamda

        Pixel = NormalizedData.shape[1]
        SampleSize=NormalizedData.shape[0]
        Flat_Data = np.reshape(NormalizedData, (SampleSize,Pixel**2))

        Linput=Loutput=Pixel**2
        Lhid = 25 
        lmb = 0 #Changed
        beta = 0.06
        rho = 0.2
        params = (Linput, Lhid, lmb, beta, rho)

        ae_lmb0=Autoencoder()
        We_lmb0=ae_lmb0.WeightInitialization(Linput,Lhid,Loutput)
        We_update_lmb0=ae_lmb0.Train(We_lmb0,Flat_Data,params,0.25,200,640)


        # In[13]:


        row = 5
        col = 5
        We_final=We_update_lmb0
        fig=plt.figure(figsize=(10, 10))
        for i in range(We_final[0].shape[1]):
            plt.subplot(row,col,i+1)
            plt.imshow(np.reshape(We_final[0][:,i],(Pixel,Pixel)), cmap='gray')
            plt.axis('off')
        plt.show()


        # In[38]:


        #Different Lamda

        Pixel = NormalizedData.shape[1]
        SampleSize=NormalizedData.shape[0]
        Flat_Data = np.reshape(NormalizedData, (SampleSize,Pixel**2))

        Linput=Loutput=Pixel**2
        Lhid = 25 
        lmb = 5e-7 #Changed
        beta = 0.06
        rho = 0.2
        params = (Linput, Lhid, lmb, beta, rho)

        ae_lmb1=Autoencoder()
        We_lmb1=ae_lmb1.WeightInitialization(Linput,Lhid,Loutput)
        We_update_lmb1=ae_lmb1.Train(We_lmb1,Flat_Data,params,0.25,200,640)


        # In[40]:


        row = 5
        col = 5
        We_final=We_update_lmb1
        fig=plt.figure(figsize=(10, 10))
        for i in range(We_final[0].shape[1]):
            plt.subplot(row,col,i+1)
            plt.imshow(np.reshape(We_final[0][:,i],(Pixel,Pixel)), cmap='gray')
            plt.axis('off')
        plt.show()


        # In[28]:


        #Different Lamda

        Pixel = NormalizedData.shape[1]
        SampleSize=NormalizedData.shape[0]
        Flat_Data = np.reshape(NormalizedData, (SampleSize,Pixel**2))

        Linput=Loutput=Pixel**2
        Lhid = 25 
        lmb = 1e-3 #Changed
        beta = 0.06
        rho = 0.2
        params = (Linput, Lhid, lmb, beta, rho)

        ae_lmb2=Autoencoder()
        We_lmb2=ae_lmb2.WeightInitialization(Linput,Lhid,Loutput)
        We_update_lmb2=ae_lmb2.Train(We_lmb2,Flat_Data,params,0.25,200,640)


        # In[41]:


        row = 5
        col = 5
        We_final=We_update_lmb2
        fig=plt.figure(figsize=(10, 10))
        for i in range(We_final[0].shape[1]):
            plt.subplot(row,col,i+1)
            plt.imshow(np.reshape(We_final[0][:,i],(Pixel,Pixel)), cmap='gray')
            plt.axis('off')
        plt.show()

    elif question == '3' :
        ##question 3 code goes here
        
        file = h5py.File(filename_Q3, "r") #Load Data

        # List all groups
        print("Keys: %s" % list(file.keys()))
        trX = list(file.keys())[0]
        trY = list(file.keys())[1]
        tstX = list(file.keys())[2]
        tstY = list(file.keys())[3]

        train_data = np.array(file[trX])
        train_labels = np.array(file[trY])
        test_data = np.array(file[tstX])
        test_labels = np.array(file[tstY])
        
   
        train_size = train_labels.shape[0]
        print("Number of Train Samples:",train_size) #Shows the number of train samples

        print("Train Data Size & Train Label Size:", train_data.shape,train_labels.shape)
        #The length of each time series is 150 units.
        #Labels: (downstairs=1, jogging=2, sitting=3, standing=4, upstairs=5, walking=6)
        
        momentum = 0.85
        l_rate = 0.1
        epoch = 30
        batch_size = 32
        RNN_Neuron=128
        indexing=np.random.permutation(train_data.shape[0])
        train_data=train_data[indexing,:,:]
        train_labels=train_labels[indexing,:]

        val_size = int(train_data.shape[0] / 10)
        val_data=train_data[:val_size,:,:]
        val_labels=train_labels[:val_size,:]
        train_data1=train_data[val_size:,:,:]
        train_labels1=train_labels[val_size:,:]

        RNN_Net = RNN_Classifier(train_data1)
        RNN_Net.addLayer(Layer(3, RNN_Neuron,'hyperbolic',1))
        RNN_Net.addLayer(Layer(RNN_Neuron,70,'relu',1))
        RNN_Net.addLayer(Layer(70,30,'relu',1))
        RNN_Net.addLayer(Layer(30,6,'softmax',1))
        crossList, TrainList = RNN_Net.TrainNetwork(l_rate,batch_size,train_data1,train_labels1,val_data,val_labels,epoch,momentum)

        plt.plot(crossList)
        plt.title('Cross-Entropy Error over Validation Set')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Error')
        plt.show()

        TestAcc=RNN_Net.Predict(test_data,test_labels)
        print("Test Accuracy: "+str(TestAcc)+"%")


        TrainAcc=RNN_Net.Predict(train_data1,train_labels1)
        print("Train Accuracy: "+str(TrainAcc)+"%")


        TestConfusion=RNN_Net.ConfusionMatrix(test_data,test_labels) 
        names = [1, 2, 3, 4, 5, 6]
        sn.heatmap(TestConfusion, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
        plt.title("Test Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Prediction")
        plt.show()


        TrainConfussion=RNN_Net.ConfusionMatrix(train_data1,train_labels1) 
        names = [1, 2, 3, 4, 5, 6]
        sn.heatmap(TrainConfussion, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
        plt.title("Train Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Prediction")
        plt.show()


        momentum = 0.85
        l_rate = 0.1
        epoch = 30
        batch_size = 32
        LSTM_Neuron=128
        indexing=np.random.permutation(train_data.shape[0])
        train_data=train_data[indexing,:,:]
        train_labels=train_labels[indexing,:]

        val_size = int(train_data.shape[0] / 10)
        val_data=train_data[:val_size,:,:]
        val_labels=train_labels[:val_size,:]
        train_data1=train_data[val_size:,:,:]
        train_labels1=train_labels[val_size:,:]

        LSTM_Net = LSTM_Classifier(train_data1)
        LSTM_Net.addLayer(LSTM_Layer(131, LSTM_Neuron,1)) #3(Sensor Values) + 128(Previous Values)
        LSTM_Net.addLayer(Layer(LSTM_Neuron,70,'relu',1))
        LSTM_Net.addLayer(Layer(70,30,'relu',1))
        LSTM_Net.addLayer(Layer(30,6,'softmax',1))
        crossList, TrainList = LSTM_Net.TrainNetwork(l_rate,batch_size,train_data1,train_labels1,val_data,val_labels,epoch,momentum)


        plt.plot(crossList)
        plt.title('Cross-Entropy Error over Validation Set')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Error')
        plt.show()

        TestAcc=LSTM_Net.Predict(test_data,test_labels)
        print("Test Accuracy: "+str(TestAcc)+"%")


        TrainAcc=LSTM_Net.Predict(train_data1,train_labels1)
        print("Train Accuracy: "+str(TrainAcc)+"%")



        TestConfusion=LSTM_Net.ConfusionMatrix(test_data,test_labels) 
        names = [1, 2, 3, 4, 5, 6]
        sn.heatmap(TestConfusion, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
        plt.title("Test Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Prediction")
        plt.show()

        
        TrainConfussion=LSTM_Net.ConfusionMatrix(train_data1,train_labels1) 
        names = [1, 2, 3, 4, 5, 6]
        sn.heatmap(TrainConfussion, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
        plt.title("Train Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Prediction")
        plt.show()


        momentum = 0.85
        l_rate = 0.1
        epoch = 30
        batch_size = 32
        GRU_Neuron=128
        indexing=np.random.permutation(train_data.shape[0])
        train_data=train_data[indexing,:,:]
        train_labels=train_labels[indexing,:]

        val_size = int(train_data.shape[0] / 10)
        val_data=train_data[:val_size,:,:]
        val_labels=train_labels[:val_size,:]
        train_data1=train_data[val_size:,:,:]
        train_labels1=train_labels[val_size:,:]

        GRU_Net = GRU_Classifier(train_data1)
        GRU_Net.addLayer(GRU_Layer(3, GRU_Neuron,1)) #3(Sensor Values) + 128(Previous Values)
        GRU_Net.addLayer(Layer(GRU_Neuron,70,'relu',1))
        GRU_Net.addLayer(Layer(70,30,'relu',1))
        GRU_Net.addLayer(Layer(30,6,'softmax',1))
        crossList, TrainList = GRU_Net.TrainNetwork(l_rate,batch_size,train_data1,train_labels1,val_data,val_labels,epoch,momentum)


        plt.plot(crossList)
        plt.title('Cross-Entropy Error over Validation Set')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Error')
        plt.show()



        TestAcc=GRU_Net.Predict(test_data,test_labels)
        print("Test Accuracy: "+str(TestAcc)+"%")


        TrainAcc=GRU_Net.Predict(train_data1,train_labels1)
        print("Train Accuracy: "+str(TrainAcc)+"%")


        # In[24]:


        TestConfusion=GRU_Net.ConfusionMatrix(test_data,test_labels) 
        names = [1, 2, 3, 4, 5, 6]
        sn.heatmap(TestConfusion, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
        plt.title("Test Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Prediction")
        plt.show()


        # In[25]:


        TrainConfussion=GRU_Net.ConfusionMatrix(train_data1,train_labels1) 
        names = [1, 2, 3, 4, 5, 6]
        sn.heatmap(TrainConfussion, annot=True, annot_kws={"size": 8}, xticklabels=names, yticklabels=names, cmap=sn.cm.rocket_r, fmt='g')
        plt.title("Train Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Prediction")
        plt.show()







#Build Class for RNN and MLP

class Layer: #Layer Class
    def __init__(self,inputDim,numNeurons,activation,beta):
        self.inputDim = inputDim
        self.numNeurons = numNeurons
        self.activation = activation
        self.beta=beta
        self.w0=np.sqrt(6/(inputDim+numNeurons))
        self.W1=np.random.uniform(-self.w0,self.w0,(inputDim,numNeurons))
        self.b1=np.random.uniform(-self.w0,self.w0,(1, numNeurons))
        self.weightsAll = np.concatenate((self.W1, self.b1), axis=0)
        self.W2=np.random.uniform(-self.w0,self.w0,(numNeurons,numNeurons))
        
        self.lastActiv=None
        self.lyrDelta=None
        self.lyrError=None
        self.prevUpdate = 0
        self.prevUpdateRNN = 0
        
    def activationFunction(self, x):
        if(self.activation == 'hyperbolic'):
            return np.tanh(self.beta*x)  
        elif(self.activation == 'softmax'):
            exp_x = np.exp(x - np.max(x))
            return exp_x/np.sum(exp_x, axis=1 ,keepdims=True)
        elif(self.activation=="relu"):
            return np.maximum(x,0)
        elif(self.activation=="sigmoid"):
            exp_x = np.exp(2*x)
            return exp_x/(1+exp_x)  
        else:
            return x
    def activationNeuron(self,x):  
        x=np.array(x)
        numSamples = x.shape[0]  
        tempInp = np.concatenate((x, -1*np.ones((numSamples, 1))), axis=1)
        self.lastActiv = self.activationFunction(np.matmul(tempInp,self.weightsAll))
        return self.lastActiv   
    
    def RecurrentActivation(self,x,hid):
        x=np.array(x)
        numSamples = x.shape[0]
        tempInp = np.concatenate((x, -1*np.ones((numSamples, 1))), axis=1)
        final=np.matmul(tempInp,self.weightsAll)+np.matmul(hid,self.W2)
        self.lastActiv = self.activationFunction(final)
        return self.lastActiv   
        
    def activation_derivative(self, x):
        if(self.activation == 'hyperbolic'):
            return self.beta*(1-(x**2)) 
        elif(self.activation == 'softmax'):
            return x*(1-x)
        elif(self.activation=="sigmoid"):
            return (x*(1-x))
        elif (self.activation=="relu"):
            return 1*(x>0)
        else:
            return np.ones(x.shape)


#Build Class for LSTM
class LSTM_Layer: #LSTM Layer Class
    def __init__(self,inputDim,numNeurons,beta):
        self.inputDim = inputDim
        self.numNeurons = numNeurons
        self.beta=beta
        self.w0=np.sqrt(6/(inputDim+numNeurons))
        # forget gate
        self.W_f=np.random.uniform(-self.w0,self.w0,(inputDim,numNeurons))
        self.bf=np.random.uniform(-self.w0,self.w0,(1, numNeurons))
        self.Wf = np.concatenate((self.W_f, self.bf), axis=0)
        # input gate
        self.W_i=np.random.uniform(-self.w0,self.w0,(inputDim,numNeurons))
        self.bi=np.random.uniform(-self.w0,self.w0,(1, numNeurons))
        self.Wi = np.concatenate((self.W_i, self.bi), axis=0)
        # cell gate
        self.W_c=np.random.uniform(-self.w0,self.w0,(inputDim,numNeurons))
        self.bc=np.random.uniform(-self.w0,self.w0,(1, numNeurons))
        self.Wc = np.concatenate((self.W_c, self.bc), axis=0)        
        # output gate
        self.W_o=np.random.uniform(-self.w0,self.w0,(inputDim,numNeurons))
        self.bo=np.random.uniform(-self.w0,self.w0,(1, numNeurons))
        self.Wo = np.concatenate((self.W_o, self.bo), axis=0) 

        
        
        self.lastActiv=None
        self.lyrDelta=None
        self.lyrError=None
        self.prevUpdate_f = 0
        self.prevUpdate_i = 0
        self.prevUpdate_c = 0
        self.prevUpdate_o = 0
        
    def activationFunction(self, x, activation):
        if(activation == 'hyperbolic'):
            return np.tanh(self.beta*x)  
        elif(activation == 'softmax'):
            exp_x = np.exp(x - np.max(x))
            return exp_x/np.sum(exp_x, axis=1)
        elif(activation=="relu"):
            return np.maximum(x,0)
        elif(activation=="sigmoid"):
            exp_x = np.exp(2*x)
            return exp_x/(1+exp_x)  
        else:
            return x
    def activationNeuron(self,x, w, activation):  
        x=np.array(x)
        numSamples = x.shape[0]  
        tempInp = np.concatenate((x, -1*np.ones((numSamples, 1))), axis=1)
        self.lastActiv = self.activationFunction(np.matmul(tempInp,w),activation)
        return self.lastActiv   
    
    def RecurrentActivation(self,x,hid, activation):
        x=np.array(x)
        numSamples = x.shape[0]
        tempInp = np.concatenate((x, -1*np.ones((numSamples, 1))), axis=1)
        final=np.matmul(tempInp,self.weightsAll)+np.matmul(hid,self.W2)
        self.lastActiv = self.activationFunction(final,activation)
        return self.lastActiv   
        
    def activation_derivative(self, x ,activation):
        if(activation == 'hyperbolic'):
            return self.beta*(1-(x**2)) 
        elif(activation == 'softmax'):
            return x*(1-x)
        elif(activation=="sigmoid"):
            return (x*(1-x))
        elif (activation=="relu"):
            return 1*(x>0)
        else:
            return np.ones(x.shape)
    

class LSTM_Classifier:
    def __init__(self,training_inputs): #CHECKED
        self.layers=[]
        self.NumSample, self.TimeSample, self.D = training_inputs.shape
        self.Hidden_prev=np.zeros((32, 128))
        self.C_prev=np.zeros((32, 128))
        
    def addLayer(self,layer): #CHECKED
        self.layers.append(layer)
    
    def FowardProp(self,training_inputs): #CHECKED
        #Foward Propagation         
        #LSTM Layer - First Layer
        lyr = self.layers[0]
        N, T, D = training_inputs.shape
        H=128        
        
        z = np.empty((N, T, D + H))
        c = np.empty((N, T, H))
        tanhc = np.empty((N, T, H))
        hf = np.empty((N, T, H))
        hi = np.empty((N, T, H))
        hc = np.empty((N, T, H))
        ho = np.empty((N, T, H))
        
        h_prev=np.zeros((N, H))
        c_prev=np.zeros((N, H))
        #Though it looks complex, just applying the functions
        for t in range(T):
            z[:, t, :] = np.column_stack((h_prev, training_inputs[:, t, :]))
            zt = z[:, t, :]
            hf[:, t, :] = lyr.activationNeuron(zt , lyr.Wf, "sigmoid")
            hi[:, t, :] = lyr.activationNeuron(zt, lyr.Wi, "sigmoid")
            hc[:, t, :] = lyr.activationNeuron(zt, lyr.Wc, "hyperbolic")
            ho[:, t, :] = lyr.activationNeuron(zt, lyr.Wo, "sigmoid")

            c[:, t, :] = hf[:, t, :] * c_prev + hi[:, t, :] * hc[:, t, :]
            
            tanhc[:, t, :] = lyr.activationFunction(c[:, t, :], "hyperbolic")
            
            h_prev = ho[:, t, :] * tanhc[:, t, :]
            c_prev = c[:, t, :]

            cache = {"z_summ": z, #Summation of h_t-1 and x_t
                     "c": c,  #Memory C_t
                     "tanhc": (tanhc), #Tanh of Memory C_t
                     "hf": hf, #Output h_f
                     "hi": (hi), #Output h_i
                     "hc": (hc),#Output h_c
                     "ho": (ho)}#Output h_o
                        

        for layer in self.layers[1:len(self.layers)]: #For MLP Layers
            h_prev=layer.activationNeuron(h_prev) 
        OUT= h_prev
        return cache,OUT

    def BackProp(self,l_rate,batch_size,training_inputs,training_labels,momentCoef):   
        cache,OUT = self.FowardProp(training_inputs)     
        foward_out = OUT
        z = cache["z_summ"]
        c=cache["c"]
        tanhc=cache["tanhc"]
        hf=cache["hf"]
        hi=cache["hi"]
        hc=cache["hc"]
        ho=cache["ho"]
        
        for i in reversed(range(len(self.layers))):# Backpropagation until LSTM
            lyr = self.layers[i]
            #outputLayer
            if(lyr == self.layers[-1]):
                lyr.lyrDelta=training_labels-foward_out    
            elif(lyr==self.layers[0]):
                nextLyr = self.layers[i+1]
                lyr.lyrError = np.matmul(nextLyr.lyrDelta, nextLyr.weightsAll[0:nextLyr.weightsAll.shape[0]-1,:].T)
                lyr.lyrDelta=lyr.lyrError
            
            else:
                nextLyr = self.layers[i+1]
                lyr.lyrError = np.matmul(nextLyr.lyrDelta, nextLyr.weightsAll[0:nextLyr.weightsAll.shape[0]-1,:].T)
                derivative=lyr.activation_derivative(lyr.lastActiv)               
                lyr.lyrDelta=derivative*lyr.lyrError
        # initialize gradients to zero
        
        dWf = 0
        dWi = 0
        dWc = 0
        dWo = 0
        H=128
        T = z.shape[1]
        NumSample, TimeSample, D = training_inputs.shape
        Prev=np.empty((NumSample, TimeSample, 128))
        
        lyr0=self.layers[0]
        delta=lyr0.lyrDelta
        #Backpropagation through time (LSTM)
        for t in reversed(range(T)):
            u = z[:, t, :]
            # if t = 0, c = 0
            if t > 0:
                c_prev = c[:, t - 1, :]
            else:
                c_prev = 0
                
            dc = delta * ho[:, t, :] * lyr0.activation_derivative(tanhc[:, t, :],"hyperbolic")
            dhf = dc * c_prev * lyr0.activation_derivative(hf[:, t, :],"sigmoid")
            dhi = dc * hc[:, t, :] * lyr0.activation_derivative(hi[:, t, :],"sigmoid")
            dhc = dc * hi[:, t, :] * lyr0.activation_derivative(hc[:, t, :],"sigmoid")
            dho = delta * tanhc[:, t, :] * lyr0.activation_derivative(ho[:, t, :],"sigmoid")

            dWf += np.matmul(np.concatenate((u, -1*np.ones((NumSample, 1))), axis=1).T, dhf)
            dWi += np.matmul(np.concatenate((u, -1*np.ones((NumSample, 1))), axis=1).T, dhi)
            dWc += np.matmul(np.concatenate((u, -1*np.ones((NumSample, 1))), axis=1).T, dhc)
            dWo += np.matmul(np.concatenate((u, -1*np.ones((NumSample, 1))), axis=1).T, dho)
            
            # update the error gradient.
            dxf = np.matmul(dhf, lyr0.Wf.T[:, :H])
            dxi = np.matmul(dhi, lyr0.Wi.T[:, :H])
            dxc = np.matmul(dhc, lyr0.Wc.T[:, :H])
            dxo = np.matmul(dho, lyr0.Wo.T[:, :H])            
            
            delta = (dxf + dxi + dxc + dxo)
            
        #Updates Weights for first and mlp layers
        for i in range(len(self.layers)):
            lyr = self.layers[i]
            if(i == 0): 
                
                update_f =  l_rate*dWf/(batch_size)
                update_i =  l_rate*dWi/(batch_size)
                update_c =  l_rate*dWc/(batch_size)
                update_o =  l_rate*dWo/(batch_size)
                
                lyr.Wf+= update_f + (momentCoef*lyr.prevUpdate_f)
                lyr.Wi+= update_i + (momentCoef*lyr.prevUpdate_i)
                lyr.Wc+= update_c + (momentCoef*lyr.prevUpdate_c)
                lyr.Wo+= update_o + (momentCoef*lyr.prevUpdate_o)
                
                lyr.prevUpdate_f = update_f 
                lyr.prevUpdate_i = update_i
                lyr.prevUpdate_c = update_c
                lyr.prevUpdate_o = update_o
                
            else:      
                numSamples = self.layers[i - 1].lastActiv.shape[0]
                tempInp=np.concatenate((self.layers[i - 1].lastActiv, -1*np.ones((numSamples, 1))), axis=1)   
                update =  l_rate*np.matmul(tempInp.T, lyr.lyrDelta)/batch_size
                lyr.weightsAll+= update + (momentCoef*lyr.prevUpdate)
                lyr.prevUpdate = update      
                
    def TrainNetwork(self,l_rate,batch_size,training_inputs,training_labels, test_inputs, test_labels, epochNum,momentCoef):
        crossList = []   
        TrainList=[]
        for epoch in range(epochNum):
            print("Epoch:",epoch)
            indexing=np.random.permutation(len(training_inputs))
            #Randomly mixing the samples
            training_inputs=training_inputs[indexing,:,:]
            training_labels=training_labels[indexing,:]
            numBatches = int(np.floor(len(training_inputs)/batch_size)) 
            for j in range(numBatches):
                train_data = training_inputs[j*batch_size:batch_size*(j+1),:,:]
                train_labels = training_labels[j*batch_size:batch_size*(j+1),:]
                self.BackProp(l_rate,batch_size,train_data,train_labels,momentCoef)         
            IN, valOutput = self.FowardProp(test_inputs)
            IN1, TrainOutput = self.FowardProp(training_inputs)
            crossErr = np.sum(-np.log(valOutput) * test_labels)/valOutput.shape[0]
            crossErr1 = np.sum(-np.log(TrainOutput) * training_labels)/TrainOutput.shape[0]
            print('Cross-Entropy Error for Validation', crossErr)
            print('Cross-Entropy Error for Train', crossErr1)
            crossList.append(crossErr)
            TrainList.append(crossErr1)
        return crossList, TrainList
    
    def Predict(self,inputs,output_real):
        Output = self.FowardProp(inputs)[1]
        Output = Output.argmax(axis=1)
        output_real = output_real.argmax(axis=1)
        return ((Output == output_real).mean()*100)
    def ConfusionMatrix(self,input1,output1):
        prediction= self.FowardProp(input1)[1]
        prediction = prediction.argmax(axis=1)
        output1 = output1.argmax(axis=1)
        K = len(np.unique(output1))
        c=np.zeros((K,K))
        for i in range(len(output1)):
            c[output1[i]][prediction[i]] += 1
        return c
    

#Build Class for GRU
class GRU_Layer: #GRU Layer Class
    def __init__(self,inputDim,numNeurons,beta):
        self.inputDim = inputDim
        self.numNeurons = numNeurons
        self.beta = beta
        self.w0=np.sqrt(6/(inputDim+numNeurons))  
        self.w1=np.sqrt(6/(numNeurons+numNeurons)) 
       
        self.W_z = np.random.uniform(-self.w0, self.w0, size=(inputDim,numNeurons))
        self.bz = np.random.uniform(-self.w0,self.w0,(1, numNeurons))
        self.Uz = np.random.uniform(-self.w1, self.w1, size=(numNeurons, numNeurons))
        self.Wz = np.concatenate((self.W_z, self.bz), axis=0) 

        self.W_r = np.random.uniform(-self.w0, self.w0, size=(inputDim,numNeurons))
        self.br = np.random.uniform(-self.w0,self.w0,(1, numNeurons))
        self.Ur = np.random.uniform(-self.w1, self.w1, size=(numNeurons, numNeurons))
        self.Wr = np.concatenate((self.W_r, self.br), axis=0) 

        self.W_h = np.random.uniform(-self.w0, self.w0, size=(inputDim,numNeurons))
        self.bh = np.random.uniform(-self.w0,self.w0,(1, numNeurons))
        self.Uh = np.random.uniform(-self.w1, self.w1, size=(numNeurons, numNeurons))
        self.Wh = np.concatenate((self.W_h, self.bh), axis=0)         
        

        self.lastActiv=None
        self.lyrDelta=None
        self.lyrError=None
        self.prevUpdate_Wz = 0
        self.prevUpdate_Wr = 0
        self.prevUpdate_Wh = 0
                                    
        self.prevUpdate_Uz = 0
        self.prevUpdate_Ur = 0
        self.prevUpdate_Uh = 0  
        
    def activationFunction(self, x, activation):
        if(activation == 'hyperbolic'):
            return np.tanh(self.beta*x)  
        elif(activation == 'softmax'):
            exp_x = np.exp(x - np.max(x))
            return exp_x/np.sum(exp_x, axis=1)
        elif(activation=="relu"):
            return np.maximum(x,0)
        elif(activation=="sigmoid"):
            exp_x = np.exp(2*x)
            return exp_x/(1+exp_x)  
        else:
            return x
    def activationNeuron(self,x, w, h, u, activation):  
        x=np.array(x)
        numSamples = x.shape[0]  
        tempInp = np.concatenate((x, -1*np.ones((numSamples, 1))), axis=1)
        self.lastActiv = self.activationFunction(np.matmul(tempInp,w)+np.matmul(h,u),activation)
        return self.lastActiv   
        
    def activation_derivative(self, x ,activation):
        if(activation == 'hyperbolic'):
            return self.beta*(1-(x**2)) 
        elif(activation == 'softmax'):
            return x*(1-x)
        elif(activation=="sigmoid"):
            return (x*(1-x))
        elif (activation=="relu"):
            return 1*(x>0)
        else:
            return np.ones(x.shape)


class GRU_Classifier:
    def __init__(self,training_inputs): #CHECKED
        self.layers=[]
        self.NumSample, self.TimeSample, self.D = training_inputs.shape
        self.Hidden_prev=np.zeros((32, 128))
        self.C_prev=np.zeros((32, 128))
        
    def addLayer(self,layer): #CHECKED
        self.layers.append(layer)
    
    def FowardProp(self,training_inputs): #CHECKED
        #Foward Propagation         
        #GRU Layer - First Layer
        lyr = self.layers[0]
        N, T, D = training_inputs.shape
        H=128        
        
        z = np.empty((N, T, H))
        r = np.empty((N, T, H))
        h_tilde = np.empty((N, T, H))
        h = np.empty((N, T, H))
        h_prev=np.zeros((N, H))
        #Similar to LSTM function. Just applying the functions.
        for t in range(T):            
            x = training_inputs[:, t, :]
            z[:, t, :] = lyr.activationNeuron(x,  lyr.Wz, h_prev, lyr.Uz , "sigmoid")
            r[:, t, :] = lyr.activationNeuron(x, lyr.Wr, h_prev,  lyr.Ur , "sigmoid")
            h_tilde[:, t, :] = lyr.activationNeuron(x, lyr.Wh, (r[:, t, :] * h_prev), lyr.Uh, "hyperbolic")
            h[:, t, :] = (1 - z[:, t, :]) * h_prev + z[:, t, :] * h_tilde[:, t, :]
            h_prev = h[:, t, :]

            cache = {"z": z, 
                     "r": r,  
                     "h_tilde": (h_tilde), 
                     "h": h}
                        

        for layer in self.layers[1:len(self.layers)]: #For MLP Layers
            h_prev=layer.activationNeuron(h_prev) 
        OUT= h_prev
        return cache,OUT

    def BackProp(self,l_rate,batch_size,training_inputs,training_labels,momentCoef):   
        cache,OUT = self.FowardProp(training_inputs)     
        foward_out = OUT
        z = cache["z"]
        r=cache["r"]
        h_tilde=cache["h_tilde"]
        h=cache["h"]

        
        for i in reversed(range(len(self.layers))):# Backpropagation until LSTM
            lyr = self.layers[i]
            #outputLayer
            if(lyr == self.layers[-1]):
                lyr.lyrDelta=training_labels-foward_out    
            elif(lyr==self.layers[0]):
                nextLyr = self.layers[i+1]
                lyr.lyrError = np.matmul(nextLyr.lyrDelta, nextLyr.weightsAll[0:nextLyr.weightsAll.shape[0]-1,:].T)
                lyr.lyrDelta=lyr.lyrError
            
            else:
                nextLyr = self.layers[i+1]
                lyr.lyrError = np.matmul(nextLyr.lyrDelta, nextLyr.weightsAll[0:nextLyr.weightsAll.shape[0]-1,:].T)
                derivative=lyr.activation_derivative(lyr.lastActiv)               
                lyr.lyrDelta=derivative*lyr.lyrError
        # initialize gradients to zero
        dWz = 0
        dUz = 0

        dWr = 0
        dUr = 0
        
        dWh = 0
        dUh = 0
        
        H=128
        NumSample, T, D = training_inputs.shape
        Prev=np.empty((NumSample, T, 128))
        
        lyr0=self.layers[0]
        delta=lyr0.lyrDelta
        #Backpropagation through time (GRU)
        for t in reversed(range(T)):
            x = training_inputs[:, t, :]
            # if t = 0, h_prev = 0, similar to LSTM
            if t > 0:
                h_prev = h[:, t - 1, :]
            else:
                h_prev = np.zeros((NumSample, H))
            
            # dE/dr is named as dr which is true for all variables (dz, dh_tilde)
            dz = delta * (h_tilde[:, t, :] - h_prev) * lyr0.activation_derivative(z[:, t, :],"sigmoid")
            dh_tilde = delta * z[:, t, :] * lyr0.activation_derivative(h_tilde[:, t, :],"hyperbolic")
            dr = (np.matmul(dh_tilde, lyr0.Uh.T) * h_prev * lyr0.activation_derivative(r[:, t, :],"sigmoid"))
            
            dWz += np.matmul(np.concatenate((x, -1*np.ones((NumSample, 1))), axis=1).T, dz)
            dUz += np.matmul(h_prev.T, dz)      

            dWr += np.matmul(np.concatenate((x, -1*np.ones((NumSample, 1))), axis=1).T, dr)
            dUr += np.matmul(h_prev.T, dr)                  

            dWh += np.matmul(np.concatenate((x, -1*np.ones((NumSample, 1))), axis=1).T, dh_tilde)
            dUh += np.matmul(h_prev.T, dh_tilde)                  
                              
            # update the error gradient.
          
            
            d1 = delta * (1 - z[:, t, :])
            d2 = np.matmul(dz, lyr0.Uz.T)
            d3 = np.matmul(dh_tilde, lyr0.Uh.T) * (r[:, t, :] + h_prev * np.matmul(lyr0.activation_derivative(r[:, t, :],"sigmoid"), lyr0.Ur.T))                  
                  
            delta = (d1+d2+d3)
            
        #Updates Weights for first and mlp layers
        for i in range(len(self.layers)):
            lyr = self.layers[i]
            if(i == 0): 
                
                update_Wz =  l_rate*dWz/(batch_size)
                update_Wr =  l_rate*dWr/(batch_size)
                update_Wh =  l_rate*dWh/(batch_size)
                update_Uz =  l_rate*dUz/(batch_size)
                update_Ur =  l_rate*dUr/(batch_size)
                update_Uh =  l_rate*dUh/(batch_size)
                  
    
        
                
                lyr.Wz+= update_Wz + (momentCoef*lyr.prevUpdate_Wz)
                lyr.Wr+= update_Wr + (momentCoef*lyr.prevUpdate_Wr)
                lyr.Wh+= update_Wh + (momentCoef*lyr.prevUpdate_Wh)
                lyr.Uz+= update_Uz + (momentCoef*lyr.prevUpdate_Uz)
                lyr.Ur+= update_Ur + (momentCoef*lyr.prevUpdate_Ur)
                lyr.Uh+= update_Uh + (momentCoef*lyr.prevUpdate_Uh)                

                  
                lyr.prevUpdate_Wz = update_Wz
                lyr.prevUpdate_Wr = update_Wr
                lyr.prevUpdate_Wh = update_Wh

                lyr.prevUpdate_Uz = update_Uz
                lyr.prevUpdate_Ur = update_Ur
                lyr.prevUpdate_Uh = update_Uh            
                  
                
            else:      
                numSamples = self.layers[i - 1].lastActiv.shape[0]
                tempInp=np.concatenate((self.layers[i - 1].lastActiv, -1*np.ones((numSamples, 1))), axis=1)   
                update =  l_rate*np.matmul(tempInp.T, lyr.lyrDelta)/batch_size
                lyr.weightsAll+= update + (momentCoef*lyr.prevUpdate)
                lyr.prevUpdate = update      
                
    def TrainNetwork(self,l_rate,batch_size,training_inputs,training_labels, test_inputs, test_labels, epochNum,momentCoef):
        crossList = []   
        TrainList=[]
        for epoch in range(epochNum):
            print("Epoch:",epoch)
            indexing=np.random.permutation(len(training_inputs))
            #Randomly mixing the samples
            training_inputs=training_inputs[indexing,:,:]
            training_labels=training_labels[indexing,:]
            numBatches = int(np.floor(len(training_inputs)/batch_size)) 
            for j in range(numBatches):
                train_data = training_inputs[j*batch_size:batch_size*(j+1),:,:]
                train_labels = training_labels[j*batch_size:batch_size*(j+1),:]
                self.BackProp(l_rate,batch_size,train_data,train_labels,momentCoef)         
            IN, valOutput = self.FowardProp(test_inputs)
            IN1, TrainOutput = self.FowardProp(training_inputs)
            crossErr = np.sum(-np.log(valOutput) * test_labels)/valOutput.shape[0]
            crossErr1 = np.sum(-np.log(TrainOutput) * training_labels)/TrainOutput.shape[0]
            print('Cross-Entropy Error for Validation', crossErr)
            print('Cross-Entropy Error for Train', crossErr1)
            crossList.append(crossErr)
            TrainList.append(crossErr1)
        return crossList, TrainList
    
    def Predict(self,inputs,output_real):
        Output = self.FowardProp(inputs)[1]
        Output = Output.argmax(axis=1)
        output_real = output_real.argmax(axis=1)
        return ((Output == output_real).mean()*100)
    def ConfusionMatrix(self,input1,output1):
        prediction= self.FowardProp(input1)[1]
        prediction = prediction.argmax(axis=1)
        output1 = output1.argmax(axis=1)
        K = len(np.unique(output1))
        c=np.zeros((K,K))
        for i in range(len(output1)):
            c[output1[i]][prediction[i]] += 1
        return c
        
    

class RNN_Classifier:
    def __init__(self,training_inputs):
        self.layers=[]
        self.NumSample, self.TimeSample, self.D = training_inputs.shape
        self.Hidden_prev=np.zeros((32, 128))
        self.RecurrentError=np.empty((32, self.TimeSample, 128))
        self.RecurrentDelta=np.empty((32, self.TimeSample, 128))
        
    def addLayer(self,layer):
        self.layers.append(layer)
    
    def FowardProp(self,training_inputs):
        #Foward Propagation
         
        #RNN Layer - First Layer
        NumSample, TimeSample, D = training_inputs.shape
        IN=np.empty((NumSample, TimeSample, 128))  
        self.Hidden_prev=np.zeros((NumSample, 128))
        for time in range(TimeSample): #You have to take into account whole time samples.
            x = training_inputs[:, time, :]
            IN[:, time, :]=self.layers[0].RecurrentActivation(x,self.Hidden_prev) 
            self.Hidden_prev=IN[:, time, :]
        OUT = IN[:, -1, :] 
            
        for layer in self.layers[1:len(self.layers)]: #For MLP Layers
            OUT=layer.activationNeuron(OUT) #Only use the last time sample as it contains memory.
        return IN,OUT

    def BackProp(self,l_rate,batch_size,training_inputs,training_labels,momentCoef):   
        IN,OUT = self.FowardProp(training_inputs)     
        foward_out = OUT
        for i in reversed(range(len(self.layers))):# Backpropagation until recurrent
            lyr = self.layers[i]
            #outputLayer
            if(lyr == self.layers[-1]):
                lyr.lyrDelta=training_labels-foward_out    
            elif (lyr == self.layers[0]):
                nextLyr = self.layers[i+1]
                lyr.lyrError = np.matmul(nextLyr.lyrDelta, nextLyr.weightsAll[0:nextLyr.weightsAll.shape[0]-1,:].T)
                self.RecurrentError[:,-1,:]=lyr.lyrError
                derivative=lyr.activation_derivative(lyr.lastActiv)               
                lyr.lyrDelta=derivative*lyr.lyrError     
                self.RecurrentDelta[:,-1,:]=lyr.lyrDelta
            else:
                nextLyr = self.layers[i+1]
                lyr.lyrError = np.matmul(nextLyr.lyrDelta, nextLyr.weightsAll[0:nextLyr.weightsAll.shape[0]-1,:].T)
                derivative=lyr.activation_derivative(lyr.lastActiv)               
                lyr.lyrDelta=derivative*lyr.lyrError
        dWall=0
        dWhid=0
        NumSample, TimeSample, D = training_inputs.shape
        Prev=np.empty((NumSample, TimeSample, 128))
        #Backpropagation through time (Recurrent)
        for time in reversed(range(TimeSample)):
            lyr=self.layers[0]
            #u = IN[:, time-1, :]
            x=training_inputs[:,time,:]
            
            if time > 0:
                u = IN[:, time-1, :]
            else:
                u = np.zeros((NumSample, 128))      
                
            derivative=lyr.activation_derivative(u)
            dWhid+=np.matmul(u.T, self.RecurrentDelta[:,time,:])            
            dWall+=np.matmul(np.concatenate((training_inputs[:,time-1,:], -1*np.ones((training_inputs[:,time-1,:].shape[0], 1))), axis=1) .T, self.RecurrentDelta[:,time,:])
            
            #For Recurrent Delta Update
            self.RecurrentError[:,time-1,:]=np.matmul(self.RecurrentDelta[:,time,:],lyr.W2.T)
            self.RecurrentDelta[:,time-1,:]=derivative*self.RecurrentError[:,time-1,:]

        #update weights
        for i in range(len(self.layers)):
            lyr = self.layers[i]
            if(i == 0):                
                update_1 =  l_rate*dWall/(batch_size*150)
                update_2 =  l_rate*dWhid/(batch_size*150)
                lyr.weightsAll+= update_1 + (momentCoef*lyr.prevUpdate)
                lyr.W2+=update_2 + (momentCoef*lyr.prevUpdateRNN)
                lyr.prevUpdate = update_1 
                lyr.prevUpdateRNN = update_2 
                
            else:      
                numSamples = self.layers[i - 1].lastActiv.shape[0]
                tempInp=np.concatenate((self.layers[i - 1].lastActiv, -1*np.ones((numSamples, 1))), axis=1)   
                update =  l_rate*np.matmul(tempInp.T, lyr.lyrDelta)/batch_size
                lyr.weightsAll+= update + (momentCoef*lyr.prevUpdate)
                lyr.prevUpdate = update      
                
    def TrainNetwork(self,l_rate,batch_size,training_inputs,training_labels, test_inputs, test_labels, epochNum,momentCoef):
        crossList = []   
        TrainList=[]
        for epoch in range(epochNum):
            print("Epoch:",epoch)
            indexing=np.random.permutation(len(training_inputs))
            #Randomly mixing the samples
            training_inputs=training_inputs[indexing,:,:]
            training_labels=training_labels[indexing,:]
            numBatches = int(np.floor(len(training_inputs)/batch_size)) 
            for j in range(numBatches):
                train_data = training_inputs[j*batch_size:batch_size*(j+1),:,:]
                train_labels = training_labels[j*batch_size:batch_size*(j+1),:]
                self.BackProp(l_rate,batch_size,train_data,train_labels,momentCoef)         
            IN, valOutput = self.FowardProp(test_inputs)
            IN1, TrainOutput = self.FowardProp(training_inputs)
            crossErr =  np.sum(-np.log(valOutput) * test_labels)/valOutput.shape[0]
            crossErr1 =  np.sum(-np.log(TrainOutput) * training_labels)/TrainOutput.shape[0]
            print('Cross-Entropy Error for Validation', crossErr)
            print('Cross-Entropy Error for Train', crossErr1)
            crossList.append(crossErr)
            TrainList.append(crossErr1)
        return crossList, TrainList
    
    def Predict(self,inputs,output_real):
        Output = self.FowardProp(inputs)[1]
        Output = Output.argmax(axis=1)
        output_real = output_real.argmax(axis=1)
        return ((Output == output_real).mean()*100)
    
    def ConfusionMatrix(self,input1,output1):
        prediction= self.FowardProp(input1)[1]
        prediction = prediction.argmax(axis=1)
        output1 = output1.argmax(axis=1)
        K = len(np.unique(output1))
        c=np.zeros((K,K))
        for i in range(len(output1)):
            c[output1[i]][prediction[i]] += 1
        return c

class Autoencoder():
    def w0(self,Lpre,Lpost): #Lpre;post are the number of neurons on either side of the connection weights.
        return np.sqrt(6/(Lpre+Lpost))
    
    def WeightInitialization(self, Lin, Lhidden, Lout):
        np.random.seed(99)
        W1=np.random.uniform(-self.w0(Lin, Lhidden),self.w0(Lin, Lhidden),(Lin,Lhidden))
        W2=np.random.uniform(-self.w0(Lhidden, Lout),self.w0(Lhidden, Lout),(Lhidden, Lout))
        b1=np.random.uniform(-self.w0(Lin, Lhidden),self.w0(Lin, Lhidden),(1, Lhidden))
        b2=np.random.uniform(-self.w0(Lhidden, Lout),self.w0(Lhidden, Lout),(1, Lout))
        We=(W1, W2, b1, b2)
        return We
    
    def sigmoid(self,x):
        #Sigmoid Function
        expx = np.exp(x)
        return expx/(1+expx)
    
    def sigmoidDerivative(self,x):
        return x*(1-x)
    
    def fowardpass(self,We,data): #Simple Foward Pass
        W1, W2, b1, b2 = We
        W1 = np.concatenate((W1,b1), axis=0)
        W2 = np.concatenate((W2,b2), axis=0)
        data_new = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)
        hid = self.sigmoid(np.matmul(data_new,W1))
        hid_new = np.concatenate((hid, np.ones((hid.shape[0], 1))), axis=1)
        out = self.sigmoid(np.matmul(hid_new,W2))
        return hid, out
    
    def aeCost(self, We, data, params):
        (Lin, Lhid, lmb, beta, rho) = params #Extra Parameters
        W1, W2, b1, b2 = We #Weights 
        N = data.shape[0] #Sample Size

        hidden, dataResult = self.fowardpass(We, data)
        hidden_mean = np.mean(hidden, axis=0)
        
        MSE = (1/(2*N))*np.sum(np.power((data - dataResult),2))
        Tyk = (lmb/2)*(np.sum(W1**2) + np.sum(W2**2))
        kl1 = rho*np.log(hidden_mean/rho)
        kl2 = (1-rho)*np.log((1-hidden_mean)/(1-rho))
        KL_final =  beta*np.sum(kl1+kl2)
        
        J = MSE + Tyk + KL_final

        deltaOut = -(data-dataResult)*self.sigmoidDerivative(dataResult)
        derKL = np.tile(beta*(-(rho/hidden_mean.T)+((1-rho)/(1-hidden_mean.T))), (N,1)).T
        deltaHid = (np.matmul(W2,deltaOut.T)+ derKL) * self.sigmoidDerivative(hidden).T
        
        gradWout = (1/N)*(np.matmul(deltaOut.T,hidden).T + lmb*W2)
        gradBout = np.mean(deltaOut, axis=0)
        gradWhid = (1/N)*(np.matmul(data.T,deltaHid.T) + lmb*W1)
        gradBhid = np.mean(deltaHid, axis=1)
        
        Jgrad=(gradWhid, gradWout, gradBhid, gradBout)
        
        return J, Jgrad
    
    def update(self, We, data, params,l_rate):
        (Lin, Lhid, lmb, beta, rho) = params #Extra Parameters
        W1, W2, b1, b2 = We #Weights 
        N = data.shape[0] #Sample Size
        J, Jgrad = self.aeCost(We, data, params)
        W1 = W1 - Jgrad[0]*l_rate
        W2 = W2 - Jgrad[1]*l_rate
        b1 = b1 - Jgrad[2]*l_rate
        b2 = b2 - Jgrad[3]*l_rate
        newWeights=(W1, W2, b1, b2)
        return J,newWeights
    
    def Train(self, We, data, params,l_rate, epochs, batch_size):
        for epoch in range(epochs):
            indexing=np.random.permutation(data.shape[0])
            data=data[indexing,:]
            numBatches = int(np.floor(data.shape[0]/batch_size))
            Total_Loss=0
            for j in range(numBatches):
                loss, We = self.update(We,data[j*batch_size:batch_size*(j+1),:],params,l_rate)
                Total_Loss=Total_Loss+loss
                
            Total_Loss=Total_Loss/numBatches
            print('Epoch', epoch+1)
            print('Loss', Total_Loss)
        return We

Atakan_Topcu_21803095_hw3(question)



