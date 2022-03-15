#This script will be used as a template for 
#finalized python file submission in EEE 443/543 Spring 2016 Course


import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import h5py
import seaborn as sn
# Atakan Topcu - 21803095

question = sys.argv[1]

def name_surname_idNumber_hw1(question):
    if question == '2' :
        print("Question",question)
        ##question 2 code goes here
         #EEE443 Assignment 1 Question 2 Part B
        hidden_Weights = np.matrix('2 0 2 4 -7; 0 -4 2 4 -5; -3 3 -2 0 -2; -2 4 0 -2 -3')
        print(hidden_Weights)
        
        inputs = np.zeros([16,5])
        i3 = -1
        i2 = -1
        i1 = -1
        i0 = -1

        for i in range(16):
            inputs[i,0] = i0
            inputs[i,1] = i1
            inputs[i,2] = i2
            inputs[i,3] = i3
            
            i3 = -i3
            if(i % 2):
                i2 = -i2
            if(i % 4 == 3):
                i1 = -i1
            if(i % 8 == 7):
                i0 = -i0
        inputs[inputs < 0] = 0
        inputs[:,4] = 1
        print(inputs) 
        
        Output_hidden=activation(inputs,hidden_Weights)
        print(Output_hidden)

        output_Weights= np.matrix('2 2 3 2 -1')
        output_bias=np.ones([16,1])
        Outp_hidden=Output_hidden.T
        Out_hidden=np.concatenate((Outp_hidden,output_bias),axis=1)
        Output=activation(Out_hidden,output_Weights)

        print("<======Output Matrix======>")
        print(Output)


        x1=inputs[:,0]
        x2=inputs[:,1]
        x3=inputs[:,2]
        x4=inputs[:,3]
        A=np.logical_or(x1,np.logical_not(x2))
        B=np.logical_or(np.logical_not(x3),np.logical_not(x4))
        Xor_list=XOR(A,B)
        boolean_output=np.array(Output, dtype=bool)
        if (boolean_output==Xor_list).all():
            print("100% Accuracy")

        

        # EEE443 Assignment 1 Question 2 Part D
                
        hiddenWeightsRobust = np.matrix('1 0 1 1 -2.5; 0 -1 1 1 -1.5; -1 1 -1 0 -0.5; -1 1 0 -1 -0.5')
        print(hiddenWeightsRobust)

        inputsRobust = np.zeros([400,5])
        for i in range(16):
            for j in range(25):
                inputsRobust[i*25+j,:] = inputs[i] #Using the inputs matrix of Part B
        print(inputsRobust.shape)

        std=0.2
        mean=0
        Noise=np.random.normal(mean, std, (400,5))
        Noise[:,4]=0 #Bias
        print(Noise.shape)
        Input_with_Noise=inputsRobust+Noise


        Output_hidden_robust=activation(Input_with_Noise,hiddenWeightsRobust)
        Output_hidden=activation(Input_with_Noise,hidden_Weights)

        output_Weights_Robust= np.matrix('1 1 1 1 -0.5')
        output_bias=np.ones([400,1])
        Outp_hidden_robust=Output_hidden_robust.T
        Out_hidden=np.concatenate((Outp_hidden_robust,output_bias),axis=1)
        Output_Robust=activation(Out_hidden,output_Weights_Robust)

        print("<======Output_Robust Matrix======>")
        print(Output_Robust)


        output_Weights= np.matrix('2 2 3 2 -1')
        output_bias=np.ones([400,1])
        Outp_hidden=Output_hidden.T
        Out_hidden=np.concatenate((Outp_hidden,output_bias),axis=1)
        Output=activation(Out_hidden,output_Weights)

        print("<======Output Matrix======>")
        print(Output)


        # In[ ]:


        x1=inputsRobust[:,0]
        x2=inputsRobust[:,1]
        x3=inputsRobust[:,2]
        x4=inputsRobust[:,3]
        A=np.logical_or(x1,np.logical_not(x2))
        B=np.logical_or(np.logical_not(x3),np.logical_not(x4))
        Xor_list=XOR(A,B)
        boolean_output=np.array(Output, dtype=bool)
        boolean_output_robust=np.array(Output_Robust, dtype=bool)


        # In[ ]:


        correct_robust=0
        correct=0
        for i in range(400):
            if(Xor_list[i] == boolean_output[0,i]):
                correct += 1
        print('Accuracy of NN in Part B = ' + str(correct/400*100) + "%")


        # In[ ]:


        for i in range(400):
            if(Xor_list[i] == boolean_output_robust[0,i]):
                correct_robust += 1
        print('Accuracy of NN in Part C = ' + str(correct_robust/400*100) + "%")


        # In[ ]:


        print("Accuracy of the NN in Part C is better " + str(correct_robust/400*100) + "% " + "> " + str(correct/400*100) + "%")



    elif question == '3' :
        print("Question",question)
        ##question 3 code goes here
        # Atakan Topcu - EEE443 Assignment 1 Q3 
        filename = "assign1_data1.h5"
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


        # In[ ]:


        #Use the first sample of each class for visualization.
        current_letter = 1
        sample_ind = list()

        row = 5
        col = 6
        fig = plt.figure()
        plt.figure()
        for i in range(train_size):
            if(current_letter == train_labels[i]):
                ax = plt.subplot(row, col, current_letter)            
                plt.imshow(train_images[:,:,i], cmap='gray')
                ax.axis('off')
                ax.autoscale(False)
                sample_ind.append(i)
                current_letter += 1
        plt.savefig('Samples.png')

        #Corr Matrix 

        num_class = 26
        corr_matrix = np.zeros(num_class**2).reshape(num_class,num_class)

        for i in range(num_class):
            for j in range(num_class):
                corr_matrix[i,j] = np.corrcoef(train_images[:,:,sample_ind[i]].flat, train_images[:,:,sample_ind[j]].flat)[0,1]

        plt.figure()      
        corrMap=sn.heatmap(corr_matrix)
        plt.savefig("CorrMatrix.png")

#Corr Matrix for within-class 

        num_class = 26
        corr_matrix = np.zeros(num_class**2).reshape(num_class,num_class)

        for i in range(num_class):
            for j in range(num_class):
                corr_matrix[i,j] = np.corrcoef(train_images[:,:,sample_ind[i]].flat, train_images[:,:,sample_ind[j]+1].flat)[0,1]
                
        corrMap=sn.heatmap(corr_matrix)
        plt.savefig("CorrMatrix_withinClass.png")

        # Part B    
        #Initialization
        initialBias=np.random.normal(0, 0.01,(num_class,1))
        pixelSize=train_images.shape[0]
        initialW=np.random.normal(0, 0.01,(num_class,pixelSize**2))

        #Arrange Labels for Sigmoid Output 
        OneHoT = np.zeros((train_labels.max().astype(int),train_labels.size))
        for i in range(train_labels.size):
            OneHoT[train_labels[i].astype(int)-1,i] = 1
        print(OneHoT)    
        MSE = list()
        iteration = 10000
        l_rate = 0.06
        for i in range(iteration):
            random_index = random.randint(0,train_size-1)
            sample=train_images[:,:,random_index].reshape(pixelSize**2,1)
            sample=sample/np.amax(sample)
            Real_Output=OneHoT[:,random_index].reshape(num_class,1)
            Output=forward(initialW,sample,initialBias)
            Difference=Real_Output-Output
            #Update
            initialW,initialBias=Update(initialW,initialBias,Difference,Output,sample,l_rate)

            MSE.append(np.sum(Difference**2)/(Difference.shape[0]))
            
        print("Sum of MSE:", sum(MSE))        
        row = 5
        col = 6
        plt.figure()
        for i in range(num_class):
            plott = plt.subplot(row, col, i+1)
            neuron_weight = initialW[i,:].reshape(pixelSize,pixelSize) 
            plt.imshow(neuron_weight, cmap='gray')
            plott.axis('off')
            plott.autoscale(True)    
            
        #plt.savefig('TrainingWeights.png')
        # Part C
        Bias_H=np.random.normal(0, 0.01,(num_class,1))
        W_H=np.random.normal(0, 0.01,(num_class,pixelSize**2))

        Bias_L=np.random.normal(0, 0.01,(num_class,1))
        W_L=np.random.normal(0, 0.01,(num_class,pixelSize**2))


        l_rate_L = 0.0002
        l_rate_H = 0.8


        MSE_H = list()
        MSE_L = list()

        iteration = 10000
        for i in range(iteration):
            random_index = random.randint(0,train_size-1)
            sample=train_images[:,:,random_index].reshape(pixelSize**2,1)
            sample=sample/np.amax(sample)
            Real_Output=OneHoT[:,random_index].reshape(num_class,1)
            Output=forward(W_L,sample,Bias_L)
            Difference=Real_Output-Output
            #Update
            W_L,Bias_L=Update(W_L,Bias_L,Difference,Output,sample,l_rate_L)

            MSE_L.append(np.sum(Difference**2)/(Difference.shape[0]))

        for i in range(iteration):
            random_index = random.randint(0,train_size-1)
            sample=train_images[:,:,random_index].reshape(pixelSize**2,1)
            sample=sample/np.amax(sample)
            Real_Output=OneHoT[:,random_index].reshape(num_class,1)
            Output=forward(W_H,sample,Bias_H)
            Difference=Real_Output-Output
            #Update
            W_H,Bias_H=Update(W_H,Bias_H,Difference,Output,sample,l_rate_H)

            MSE_H.append(np.sum(Difference**2)/(Difference.shape[0]))        

        plt.figure()
        plt.plot(MSE_H)
        plt.plot(MSE_L)
        plt.plot(MSE)
        plt.legend(["MSE for u="+str(l_rate_H), "MSE for u="+str(l_rate_L), "MSE for u="+str(l_rate)])
        plt.title("Mean Squared Errors for Different Learning Rates")
        plt.xlabel("Iteration Number")
        plt.ylabel("MSE")
        #plt.savefig('MSE_Changes.png')
        plt.show()
        # Part D
        print("Test Image Size & Test Label Size:", test_images.shape,test_labels.shape)

        test_images = test_images.reshape(pixelSize**2,test_labels.shape[0])
        test_size = test_labels.shape[0]
        bias_matrix = np.zeros((num_class, test_size))

        for i in range(test_size):
            bias_matrix[:,i] = initialBias.flatten()
        print("bias_matrix size:" ,bias_matrix.shape) 

        test_images = test_images/np.amax(test_images)
        Output = forward(initialW,test_images,bias_matrix)
        print("Output Matrix Size:",Output.shape)

        #For learning rate = 0.06
        Output_indices = np.zeros(Output.shape[1])
        for i in range(Output.shape[1]):
            Output_indices[i] = np.argmax(Output[:,i])+1 #Returns the index of the maximum element, add 1 since index starts from 0

        true_count = 0
        for i in range(Output_indices.shape[0]):
            if(Output_indices[i] == test_labels[i]):
                true_count += 1;
                
        print('Accuracy for Learning rate = 0.06: ', round(true_count/test_labels.shape[0]*100,2),"%")

        #For learning rate = 0.8
        bias_matrix_H= np.zeros((num_class, test_size))

        for i in range(test_size):
            bias_matrix_H[:,i] = Bias_H.flatten()
        print("bias_matrix size:" ,bias_matrix_H.shape) 

        Output_H = forward(W_H,test_images,bias_matrix_H)

        Output_indices = np.zeros(Output_H.shape[1])

        for i in range(Output_H.shape[1]):
            Output_indices[i] = np.argmax(Output_H[:,i])+1 #Returns the index of the maximum element, add 1 since index starts from 0

        true_count = 0
        for i in range(Output_indices.shape[0]):
            if(Output_indices[i] == test_labels[i]):
                true_count += 1;
                
        print('Accuracy for Learning rate = 0.8: ', round(true_count/test_labels.shape[0]*100,2),"%")

        #For learning rate = 0.0002
        bias_matrix_L= np.zeros((num_class, test_size))

        for i in range(test_size):
            bias_matrix_L[:,i] = Bias_L.flatten()
        print("bias_matrix size:" ,bias_matrix_L.shape) 

        Output_L = forward(W_L,test_images,bias_matrix_L)

        Output_indices = np.zeros(Output_L.shape[1])
        for i in range(Output_L.shape[1]):
            Output_indices[i] = np.argmax(Output_L[:,i])+1 #Returns the index of the maximum element, add 1 since index starts from 0

        true_count = 0
        for i in range(Output_indices.shape[0]):
            if(Output_indices[i] == test_labels[i]):
                true_count += 1;
                
        print('Accuracy for Learning rate = 0.0002: ', round(true_count/test_labels.shape[0]*100,2),"%")

#End Here





#Now we will define the activation function (i.e, step function)
def activation(inputs,weights):
    v=np.matmul(weights,inputs.T)
    v[v >= 0] = 1
    v[v < 0]= 0
    output = v
    return output


def XOR(A,B):
    boolean_arrayA=np.array(A, dtype=bool)
    boolean_arrayB=np.array(B, dtype=bool)
    Part1=np.logical_and(boolean_arrayA,np.logical_not(boolean_arrayB))
    Part2=np.logical_and(np.logical_not(boolean_arrayA),boolean_arrayB)
    Out=np.logical_or(Part1,Part2)
    return Out

def sigmoid(activation_output):
    transfer=1/(1 + np.exp(-activation_output))
    return transfer

def forward(W, x, b):
    return sigmoid(np.matmul(W,x) - b)

def Update(W,Bias,Difference,Output,sample,l_rate):
    w_grd = -2*np.matmul(Difference*(Output)*(1-Output),np.transpose(sample))
    b_grd = 2*Difference*(Output)*(1-Output)    
            
    W -= l_rate*w_grd
    Bias -= l_rate*b_grd    
    return W,Bias

name_surname_idNumber_hw1(question)