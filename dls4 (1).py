import numpy as np
import pandas as pd
from math import sqrt
from scipy import optimize as op
from matplotlib import pyplot as plt
from scipy.optimize import nnls
from scipy.stats import norm, laplace



############  DATA PRE-PROCESSING  ##################

main_data = pd.read_csv('04_cricket_1999to2011.csv',sep=',')

data = main_data[main_data['Innings']==1]
data = data[['Match','Over','Runs','Total.Out']]
data = np.array(data)
# Nx100 Matrix holding history data 
Innings1 = list()

inning = np.zeros(100)
i=0
for d in data[:]:
    if d[1]==1 or i==50:
        while i!=50:
            inning[i],inning[i+50] = inning[i-1],inning[50+i-1]
            i+=1
        if inning[49]<400 and inning[49]>150 and inning[50]>=0  and inning[-1]<=10 and np.all(np.diff(inning[:50])>=0) and np.all(np.diff(inning[50:])>=0):
            Innings1.append(inning)
        inning = np.zeros(100)
        i=0
    inning[i],inning[i+50] = inning[i-1]+d[2],d[3]
    i+=1

Innings1 = np.array(Innings1)     
Innings1 = Innings1[:,:]


#The over at which 10 wickets fall in innings 1
Innings1_allOut = np.zeros(Innings1.shape[0])
i=0
for inning in Innings1:
    Innings1_allOut[i] = np.searchsorted(inning[50:],10)
    i+=1


data = main_data[main_data['Innings']==2]
data = data[['Over','Runs','Total.Out','Target.Score']]
data = np.array(data)
# Nx101 Matrix with Target in last column
Innings2 = list()

inning = np.zeros(101)
i=0
target=0
for d in data[:]:
    if d[1]==1 or i==50:
        while i!=50:
            inning[i],inning[i+50] = inning[i-1],inning[50+i-1]
            i+=1
        if inning[49]<400 and inning[49]>150 and inning[50]>=0  and inning[-1]<=10 and np.all(np.diff(inning[:50])>=0) and np.all(np.diff(inning[50:])>=0):
            inning[-1] = target
            Innings2.append(inning)
        inning = np.zeros(100)
        i=0
        target = d[3]
    inning[i],inning[i+50] = inning[i-1]+d[1],d[2]
    i+=1

Innings2 = np.array(Innings2)     
Innings2 = Innings2[:,:]

#The over at which 10 wickets fall in innings 2
Innings2_allOut = np.zeros(Innings2.shape[0])
i=0
for inning in Innings2:
    Innings2_allOut[i] = np.searchsorted(inning[50:],10)
    i+=1


########################################################





########################   mRSC Algorithm   ###############################

def mRSC(X, data, intervention_point, k):
    N=5
    params = list(np.zeros(len(data)))
    
    # SVD decomposition
    U,s,V = np.linalg.svd(data)
    X = X.reshape((-1,1))

    # SVD reconstruction
    S = np.zeros(np.shape(data))
    for i in range(0, k):
        S[i,i] = s[i]
    M = U @ S @ V

    #print(X.shape)

    # Now we create the truncated smoothed data matrix with respect to which we will perform the regression
    M_tr = np.zeros(shape=(M.shape[0], 2*intervention_point))
    M_tr[:,:intervention_point],M_tr[:,intervention_point:] = M[:,:intervention_point],M[:,50:50+intervention_point]

    #print(X.flatten().shape)
    
    popt = nnls(M_tr.T, X.flatten())
    ls_error = popt[1]
    popt = np.reshape(popt[0], (-1,1))
    result = np.int16(M.T@popt)
    
    return result, ls_error


###########################################################################




"""TASKS"""

# vary k and see how the errors stack up i.e average over all 200 matches various values of k = [1,2,3,4,5,10,25]
# vary intervention_point per over
# vary k and intervention_point as k=[5,10,15] and intervention_point=[5,10,15,20,25,30,35,40,45] and then plot a scatter plot where the y-axis is error (x_axis doesn't signify anything)
# MAKE SURE TO LABEL THE AXES AND TITLE THE GRAPHS CORRECTLY.
# For saving pdf a particular experiment/graph just change the name. And the numbering has been taken care of.
# NOTE: You have to take care of cases when matches end before 50 0vers i.e intervention point will now go only upto the matches till which the game has been played.



#################################   Data Fitting and Graph Generation for innings 1  #############################################

def data_fitting(train_set, cv_set, end_overs, intervention_point=30, k=10, plot = False, plots=6, name = 'xyz'):
    errors = []
    ls_error = []
    samples = [np.random.randint(1, len(cv_set)) for _ in range(plots)]
    print(samples)
    j = 0
    for innings in cv_set:
        
        X = np.zeros(2*intervention_point)
                
        X[:intervention_point],X[intervention_point:] = innings[:intervention_point],innings[50:50+intervention_point]
        result, ls = mRSC(X, train_set, intervention_point, k)
        result = result.flatten()

        #************************************************** Make sure this is correct
        # Final Projected Score - Actual Final Score
        errors.append(result[end_overs[j]]-innings[end_overs[j]])
        j += 1
        
        """
        print(ls, result[-1]-innings[-1])
        print()
        print("Predicted Trajectory")
        print(result)
        print("Actual Trajectory")
        print(innings)
        print("Maximum Run/Wicket Deviation")
        print(np.max(np.abs(innings - result)))
        print('#####################################')
        print('#####################################')
        ls_error.append(ls)"""

        if plot and (j in samples):
            i = 1
            end = end_overs[j]
            plt.plot(range(end), result[0:end], label = 'Predicted Score')
            plt.plot(range(end), innings[0:end], label = 'Actual Score')
            plt.title('Sample Trajectory Prediction ' + str(i))
            plt.legend(loc="lower right")
            plt.xlabel('Overs')
            plt.ylabel('Cumulative Runs')
            i+=1
            name = name + 'j'
            plt.savefig(name +".pdf", dpi=200)
            plt.show()
            
            print(ls, result[-1]-innings[-1])
            print()
            print("Predicted Trajectory")
            print(result)
            print("Actual Trajectory")
            print(innings)
            print("Maximum Run/Wicket Deviation")
            print(np.max(np.abs(innings - result)))
            print('#####################################')
            print('#####################################')
            ls_error.append(ls)
            
        
    return errors


# The perm will hold the particular permuttion of the Innings1
perm = np.random.permutation(1400)

Innings1 = Innings1[0:1400]
Innings1 = Innings1[perm]

Innings2 = Innings2[0:1400]
Innings2 = Innings2[perm]

train_samples = 1000
cv_samples = 200

train_set = Innings1[0:train_samples]
cv_set = Innings1[train_samples:train_samples+cv_samples]


data_fitting(train_set, cv_set, end_overs = Innings1_allOut[train_samples:train_samples+cv_samples], intervention_point=30, k=10, plot = True, plots=2)







"""TASKS"""

# Somehow just obtain the error matrix



#**************** Need to write a code to get back the errors ******************************************


#################################   Data Fitting and Error Probability curve generation for Innings 2  #############################################



def data_fitting(train_set, cv_set, end_overs, intervention_point=30, k=10, plot = False, plots=6, nanme = 'xyz'):
    errors = []
    ls_error = []
    samples = [np.random.randint(1, len(cv_set)) for _ in range(plots)]
    print(samples)
    j = 0
    for innings in cv_set:
        
        X = np.zeros(2*intervention_point)
                
        X[:intervention_point],X[intervention_point:] = innings[:intervention_point],innings[50:50+intervention_point]
        result, ls = mRSC(X, train_set, intervention_point, k)
        result = result.flatten()

        #************************************************** Make sure this is correct
        # Final Projected Score - Actual Final Score
        errors.append(result[-1]-innings[-1])
        j += 1
        
        """
        print(ls, result[-1]-innings[-1])
        print()
        print("Predicted Trajectory")
        print(result)
        print("Actual Trajectory")
        print(innings)
        print("Maximum Run/Wicket Deviation")
        print(np.max(np.abs(innings - result)))
        print('#####################################')
        print('#####################################')
        ls_error.append(ls)"""

        if plot and (j in samples):
            i = 1
            end = end_overs[j]
            plt.plot(range(end), result[0:end], label = 'Predicted Score')
            plt.plot(range(end), innings[0:end], label = 'Actual Score')
            plt.title('Sample Trajectory Prediction ' + str(i))
            plt.legend(loc="lower right")
            plt.xlabel('Overs')
            plt.ylabel('Cumulative Runs')
            i+=1
            name = name + 'j'
            plt.savefig(name +".pdf", dpi=200)
            plt.show()
            
            print(ls, result[-1]-innings[-1])
            print()
            print("Predicted Trajectory")
            print(result)
            print("Actual Trajectory")
            print(innings)
            print("Maximum Run/Wicket Deviation")
            print(np.max(np.abs(innings - result)))
            print('#####################################')
            print('#####################################')
            ls_error.append(ls)
            
        
    return errors







# The perm will hold the particular permuttion of the Innings1
perm = np.random.permutation(1400)

Innings1 = Innings1[0:1400]
Innings1 = Innings1[perm]

Innings2 = Innings2[0:1400]
Innings2 = Innings2[perm]

train_samples = 800
cv_samples = 300
test_samples = 300

train_set = Innings1[0:train_samples]
cv_set = Innings1[train_samples:train_samples+cv_samples]
test_set = Innings1[train_samples+cv_samples:1400]

errors = data_fitting(train_set, cv_set, end_overs = Innings2_allOut[train_samples:train_samples+cv_samples], intervention_point=30, k=10, plot = True, plots=2)












# This function takes in the following 3 arguments: The sample_errors obtained from the Cross Validation set after comparing predicted scores to target scores
# The targets set by the first team in the match
# The predicted_scores on the test set
# Finally the actual_scores scored by team 2 in the match to check whether our prediction is correct


def prediction_confidence(sample_errors, predicted_scores, targets, actual_scores, distribution = 'gaussian'):

    # A positive comfort margin implies your predicted score is above the target, while negative implies it is lesser
    comfort_margin = np.array(predicted_scores) - np.array(targets)

    prediction = np.zeros(len(targets))
    
    if distribution == 'gaussian':
        mu = np.mean(np.array(sample_errors))
        sigma = np.std(np.array(sample_errors))
        cdf = norm.cdf(-comfort_margin, loc=mu, scale = sigma)
        confidence = 1-cdf
        prediction[confidence>1] = 1
        

    elif distribution == 'laplacian':
        mu,sigma = laplace.fit(np.array(sample_errors))
        cdf = laplace.cdf(-comfort_margin, loc=mu, scale = sigma)
        confidence = 1-cdf
        prediction[confidence>1] = 1


    # Calculates percentage accuracy our deterministic prediction i.e if winning confidence > 0.5 then outcome=1 and its reverse is 
    predicted_outcomes = np.zeros(len(targets))
    actual_outcomes = np.zeros(len(targets))
    predicted_outcome[confidence>0.5] = 1
    actual_outcomes[(actual_scores-target) > 0] = 1

    correct_pred = np.sum(predicted_outcome==actual_outcomes, dtype = np.int16)
    percent = correct_pred/len(targets)
    
    
    # Note that the confidence is the confidence of Team 2 winning and NOT the confidence of the prediction
    return prediction, confidence, percent






