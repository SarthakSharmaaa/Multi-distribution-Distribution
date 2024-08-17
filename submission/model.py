import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl


class NaiveBayes:
    
    def __init__(self):
        
        self.priors = {}
        self.guassian = {}
        self.bernoulli = {}
        self.laplace = {}
        self.exponential = {}
        self.multinomial = {}
        
        
    def fit(self, X, y):

        """Start of your code."""
        """
        X : np.array of shape (n,10)
        y : np.array of shape (n,)
        Create a variable to store number of unique classes in the dataset.
        Assume Prior for each class to be ratio of number of data points in that class to total number of data points.
        Fit a distribution for each feature for each class.
        Store the parameters of the distribution in suitable data structure, for example you could create a class for each distribution and store the parameters in the class object.
        You can create a separate function for fitting each distribution in its and call it here.
        """
        

        """Start your code"""
        tot_datapoints=len(y)
        classes=np.unique(y)
        
        data = [ [] for i in range(3)]
        
        for _x,_y in zip(X,y):
            data[int(_y)].append(_x)
            
        data = np.array(data)
        
        
            
        for i in range(3):
            data1 = data[i][:,0]
            data2 = data[i][:,1]
            data3 = data[i][:,2]
            data4 = data[i][:,3]
            data5 = data[i][:,4]
            data6 = data[i][:,5]
            data7 = data[i][:,6]
            data8 = data[i][:,7]
            data9 = data[i][:,8]
            data10 = data[i][:,9]
            
            self.priors[str(i)]=np.sum(y==i)/tot_datapoints
            
            mean1=np.mean(data1,axis=0)
            variance1=np.var(data1,axis=0)
            mean2=np.mean(data2,axis=0)
            variance2=np.var(data2,axis=0)
            self.guassian[str(i)]=[mean1,mean2,variance1,variance2]
            
            p1=np.mean(data3,axis=0)
            p2=np.mean(data4,axis=0)
            self.bernoulli[str(i)]=[p1,p2]
            
            mu1= np.mean(data5, axis=0)
            b1= np.mean(np.abs(data5 - mu1), axis=0)
            mu2= np.mean(data6, axis=0)
            b2= np.mean(np.abs(data6 - mu2), axis=0)
            self.laplace[str(i)]=[mu1,mu2,b1,b2]
            
            lambda1=1/np.mean(data7,axis=0)
            lambda2=1/np.mean(data8,axis=0)
            self.exponential[str(i)]=[lambda1,lambda2]
            
            ct9 = int(max(data9)) + 1
            mean9 = [0 for i in range(ct9)]
            for x in data9:
                mean9[int(x)] += 1
            mean9 = [x/len(data9) for x in mean9]    
            
            
            ct10 = int(max(data10)) + 1
            mean10 = [0 for i in range(ct10)]
            for x in data10:
                mean10[int(x)] += 1
            mean10 = [x/len(data10) for x in mean10]
            
            self.multinomial[str(i)] = [mean9, mean10]    
            
            
        """End your code"""
        
        
        


        """End of your code."""

    def get_gaussian(self,mean,variance,x):
        coefficient = 1 / np.sqrt(2 * np.pi * variance)
        exponent = -((x - mean) ** 2) / (2 * variance)
        return coefficient * np.exp(exponent)
    
    def get_exponent(self,lambdaval,x):
        return lambdaval * np.exp(-lambdaval * x)
    
    def get_laplacian(self,mu,b,x):
        return 1/(2*b) * np.exp(-np.abs(x - mu) / b)
    
    def get_bernoulli(self,p,x):
        return p**x * (1-p)**(1-x)
    
    def get_multinomial(self, p_values, x, y):
        return p_values[int(x)]
        
    def predict(self, X):
        """Start of your code."""
        """
        X : np.array of shape (n,10)

        Calculate the posterior probability using the parameters of the distribution calculated in fit function.
        Take care of underflow errors suitably (Hint: Take log of probabilities)
        Return an np.array() of predictions where predictions[i] is the predicted class for ith data point in X.
        It is implied that prediction[i] is the class that maximizes posterior probability for ith data point in X.
        You can create a separate function for calculating posterior probability and call it here.
        """
        predicted=[]
        for x in X:
            logarr=self.probability(x)
            predicted_class=np.argmax(logarr)
            predicted.append(predicted_class)
            
        return np.array(predicted)
    
    def probability(self,x):
        logarr=[]
        for i in range(3):
            logprob=np.log(self.priors[str(i)])
            
            gauss1=np.log(self.get_gaussian(self.guassian[str(i)][0],self.guassian[str(i)][2],x[0]))
            gauss2=np.log(self.get_gaussian(self.guassian[str(i)][1],self.guassian[str(i)][3],x[1]))
            logprob=logprob+gauss1+gauss2
            
            berno1=np.log(self.get_bernoulli(self.bernoulli[str(i)][0],x[2]))
            berno2=np.log(self.get_bernoulli(self.bernoulli[str(i)][1],x[3]))
            logprob=logprob+berno1+berno2
            
            laplace1=np.log(self.get_laplacian(self.laplace[str(i)][0],self.laplace[str(i)][2],x[4]))
            laplace2=np.log(self.get_laplacian(self.laplace[str(i)][1],self.laplace[str(i)][3],x[5]))
            logprob=logprob+laplace1+laplace2
            
            expo1=np.log(self.get_exponent(self.exponential[str(i)][0],x[6]))
            expo2=np.log(self.get_exponent(self.exponential[str(i)][1],x[7]))
            logprob=logprob+expo1+expo2
            
            multi1=np.log(self.get_multinomial(self.multinomial[str(i)][0],x[8],i))
            multi2=np.log(self.get_multinomial(self.multinomial[str(i)][1],x[9],i))
            logprob=logprob+multi1+multi2
            
            
            
            logarr.append(logprob)
        return np.array(logarr)
        

    def getParams(self):
        """
        Return your calculated priors and parameters for all the classes in the form of dictionary that will be used for evaluation
        Please don't change the dictionary names
        Here is what the output would look like:
        priors = {"0":0.2,"1":0.3,"2":0.5}
        gaussian = {"0":[mean_x1,mean_x2,var_x1,var_x2],"1":[mean_x1,mean_x2,var_x1,var_x2],"2":[mean_x1,mean_x2,var_x1,var_x2]}
        bernoulli = {"0":[p_x3,p_x4],"1":[p_x3,p_x4],"2":[p_x3,p_x4]}
        laplace = {"0":[mu_x5,mu_x6,b_x5,b_x6],"1":[mu_x5,mu_x6,b_x5,b_x6],"2":[mu_x5,mu_x6,b_x5,b_x6]}
        exponential = {"0":[lambda_x7,lambda_x8],"1":[lambda_x7,lambda_x8],"2":[lambda_x7,lambda_x8]}
        multinomial = {"0":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"1":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"2":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]]}
        """
        return (self.priors,self.guassian,self.bernoulli,self.laplace,self.exponential,self.multinomial)   


def save_model(model,filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open("model.pkl","wb")
    pkl.dump(model,file)
    file.close()

def load_model(filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open(filename,"rb")
    model = pkl.load(file)
    file.close()
    return model

def visualise(data_points,labels):
    """
    datapoints: np.array of shape (n,2)
    labels: np.array of shape (n,)
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('Generated 2D Data from 5 Gaussian Distributions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def net_f1score(predictions, true_labels):
    """Calculate the multclass f1 score of the predictions.
    For this, we calculate the f1-score for each class 

    Args:
        predictions (np.array): The predicted labels.
        true_labels (np.array): The true labels.

    Returns:
        float(list): The f1 score of the predictions for each class
    """

    def precision(predictions, true_labels, label):
        """Calculate the multclass precision of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The precision of the predictions.
        """
        """Start of your code."""
        true_positives = np.sum((predictions == label) & (true_labels == label))
        false_positives = np.sum((predictions == label) & (true_labels != label))
        precision = true_positives / (true_positives + false_positives)
        return precision


        
        """End of your code."""
        


    def recall(predictions, true_labels, label):
        """Calculate the multclass recall of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.
        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The recall of the predictions.
        """
        """Start of your code."""
        true_positives = np.sum((predictions == label) & (true_labels == label))
        false_negatives = np.sum((predictions != label) & (true_labels == label))
        recall = true_positives / (true_positives + false_negatives)
        return recall


        """End of your code."""
        

    def f1score(predictions, true_labels, label):
        """Calculate the f1 score using it's relation with precision and recall.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The f1 score of the predictions.
        """

        """Start of your code."""
        prec = precision(predictions, true_labels, label)
        rec = recall(predictions, true_labels, label)
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
        return f1

        """End of your code."""
    

    f1s = []
    for label in np.unique(true_labels):
        f1s.append(f1score(predictions, true_labels, label))
    return f1s

def accuracy(predictions,true_labels):
    """

    You are not required to modify this part of the code.

    """
    return np.sum(predictions==true_labels)/predictions.size



if __name__ == "__main__":
    """

    You are not required to modify this part of the code.

    """

    # Load the data
    train_dataset = pd.read_csv('./data/train_dataset.csv',index_col=0).to_numpy()
    validation_dataset = pd.read_csv('./data/validation_dataset.csv',index_col=0).to_numpy()

    # Extract the data
    train_datapoints = train_dataset[:,:-1]
    train_labels = train_dataset[:, -1]
    validation_datapoints = validation_dataset[:, 0:-1]
    validation_labels = validation_dataset[:, -1]

    # Visualize the data
    # visualise(train_datapoints, train_labels, "train_data.png")

    # Train the model
    model = NaiveBayes()
    model.fit(train_datapoints, train_labels)

    # Make predictions
    train_predictions = model.predict(train_datapoints)
    validation_predictions = model.predict(validation_datapoints)

    # Calculate the accuracy
    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(validation_predictions, validation_labels)

    # Calculate the f1 score
    train_f1score = net_f1score(train_predictions, train_labels)
    validation_f1score = net_f1score(validation_predictions, validation_labels)

    # Print the results
    print('Training Accuracy: ', train_accuracy)
    print('Validation Accuracy: ', validation_accuracy)
    print('Training F1 Score: ', train_f1score)
    print('Validation F1 Score: ', validation_f1score)

    # Save the model
    save_model(model)

    # Visualize the predictions
    # visualise(validation_datapoints, validation_predictions, "validation_predictions.png")

