from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv


total_data_size = 506
training_set_size = 433
testing_set_size = 73
feature_count = 13

feature_names = ["bias","CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]

def getMSE(X,Y,W):
	return float(np.dot(np.transpose(np.subtract(Y,np.dot(X,W))),np.subtract(Y,np.dot(X,W))))/len(Y)


def getW(X,Y):
	return np.dot(np.dot(pinv(np.dot(np.transpose(X),X)),np.transpose(X)),Y)


def getRRW(X,Y,lam):
	identityM = len(Y)*lam*np.identity(14)
	identityM[0][0] = 0
	return np.dot(np.dot(pinv(np.add(identityM,np.dot(np.transpose(X),X))),np.transpose(X)),Y)

def standardizeSet(data_set,mu,sigma):
	std_data_set = data_set
	for rowIdx in range(0,len(data_set)):
		for columnIdx in range(1,len(data_set[rowIdx])):
			std_data_set[rowIdx][columnIdx] = float(data_set[rowIdx][columnIdx] - mu[columnIdx])/sigma[columnIdx]
	return std_data_set


def getStandardizeParams(training_set_X):
	mu = [0.0 for i in range(0,training_set_X[0].size)]
	sigma = [0.0 for i in range(0,training_set_X[0].size)]
	for columnIdx in range(1,training_set_X[0].size):
		z = column(training_set_X,columnIdx)
		mu[columnIdx] = float(sum(z))/len(z)
		for i in range(0,len(z)):
			sigma[columnIdx] += (training_set_X[i][columnIdx] - mu[columnIdx])**2
		sigma[columnIdx] = float(sigma[columnIdx])/(len(z))
		sigma[columnIdx] = sigma[columnIdx]**(0.5)
	return mu,sigma

def column(array,index):
	return [row[index] for row in array]

def plotHistogram(training_set_X):
	for columnIdx in range(1,feature_count+1):
		z = column(training_set_X,columnIdx)
		plt.hist(z,bins=10)
		plt.suptitle(feature_names[columnIdx])
		plt.show()

def makeTrainNTest(boston):
	training_set_X = []
	testing_set_X = []
	training_set_Y = []
	testing_set_Y = []
	for i in range(0,total_data_size):
		if i%7 == 0:
			testing_set_X.append(np.append([1],boston.data[i]))
			testing_set_Y.append(boston.target[i])
		else:
			training_set_X.append(np.append([1],boston.data[i]))
			training_set_Y.append(boston.target[i])
	return training_set_X,training_set_Y,testing_set_X,testing_set_Y


def crossValidation(std_training_set_X,training_set_Y,std_testing_set_X,testing_set_Y):
	print "-------------------k-fold cross validation-------------------"
	lam = 0.0001
	best_lam = 0.0001
	least_error = float('inf')
	srno = 1
	print "lambda updated by factor of 10^0.0625 or 1.15478198469"

	print "SR.\tlambda\t\tCV error"
	while(lam <= 10):
		CVerror = [0 for i in range(0,10)]
		for k in range(1,10):
			k_fold_train_set_X = []
			k_fold_train_set_Y = []
			k_fold_test_set_X = []
			k_fold_test_set_Y = []
			for i in range(0,len(training_set_Y)):
				if i in range(k*len(training_set_Y)/10,(k+1)*len(training_set_Y)/10):
					k_fold_test_set_X.append(std_training_set_X[i])
					k_fold_test_set_Y.append(training_set_Y[i])
				else:
					k_fold_train_set_X.append(std_training_set_X[i])
					k_fold_train_set_Y.append(training_set_Y[i])
			k_fold_test_set_X = np.array(k_fold_test_set_X)
			k_fold_train_set_X = np.array(k_fold_train_set_X)
			k_fold_test_set_Y = np.array(k_fold_test_set_Y)
			k_fold_train_set_Y = np.array(k_fold_train_set_Y)		
			RRW = getRRW(k_fold_train_set_X,k_fold_train_set_Y,lam)
			CVerror[k] = getMSE(k_fold_test_set_X,k_fold_test_set_Y,RRW)
		current_error = np.average(CVerror)
		if least_error > current_error:
			least_error = current_error
			best_lam = lam		
		print srno,"\t",format(lam,'.6f'),"\t",format(current_error,'.6f')
		lam = lam*1.1547819846 #10^0.0625
		srno += 1

	print "best lambda is",format(best_lam,'.6f'), "with CV error",format(least_error,'.6f')
	RRW_best_lambda = getRRW(std_training_set_X,training_set_Y,best_lam)
	print "MSE for training set (lambda=",format(best_lam,'.6f'),") is",getMSE(std_training_set_X,training_set_Y,RRW_best_lambda)
	print "MSE for testing set (lambda=",format(best_lam,'.6f'),") is",getMSE(std_testing_set_X,testing_set_Y,RRW_best_lambda)
	return best_lam



def getCorrelation(X,Y):
	correlation = [0 for i in range(0,feature_count+1)]
	for i in range(1,feature_count+1):
		Xcol = np.array(column(X,i))
		Y = np.array(Y)
		r_num = len(Y)*(np.dot(np.transpose(Y),Xcol)) - (np.sum(Xcol)*np.sum(Y))
		r_den1 = np.sqrt(len(Y)*(np.dot(np.transpose(Xcol),Xcol)) - (np.sum(Xcol)*np.sum(Xcol)))
		r_den2 = np.sqrt(len(Y)*(np.dot(np.transpose(Y),Y)) - (np.sum(Y)*np.sum(Y)))
		correlation[i] = r_num/(r_den1 * r_den2)
	return correlation


def getBruteForceList(depth,X,Y,selected_features,tentative_list):
	if depth == 4:
		if len(selected_features) == 4:
			#print selected_features
			tentative_list.append(selected_features)
	elif depth == 0:
		for i in range(1,14):
			new_selected_features = np.append(selected_features,[i])
			tentative_list = getBruteForceList(depth+1,X,Y,new_selected_features,tentative_list)
	else:
		startIdx = selected_features[len(selected_features)-1]+1
		for i in range(startIdx,14):
			new_selected_features = np.append(selected_features,i)
			tentative_list = getBruteForceList(depth+1,X,Y,new_selected_features,tentative_list)
	return tentative_list
				
def main():
	boston = load_boston()
	training_set_X,training_set_Y,testing_set_X,testing_set_Y = makeTrainNTest(boston)
	plotHistogram(training_set_X)
	correlation = getCorrelation(training_set_X,training_set_Y)
	abs_correlation = map(abs,correlation)
	print"-------------------Pearson correlation-------------------"
	print "feature no.\tPearson correlation\tAbsolute correlation"
	for i in range(1,feature_count+1):
		print i,"\t\t",format(correlation[i],'.6f'),"\t\t",format(abs_correlation[i],'.6f')
	mu,sigma = getStandardizeParams(training_set_X)
	std_training_set_X = np.array(standardizeSet(training_set_X,mu,sigma))
	std_testing_set_X = np.array(standardizeSet(testing_set_X,mu,sigma))
	training_set_Y = np.array(training_set_Y)
	testing_set_Y = np.array(testing_set_Y)	
	W = getW(std_training_set_X,training_set_Y)
	
	RRW001 = getRRW(std_training_set_X,training_set_Y,0.01)
	RRW01 = getRRW(std_training_set_X,training_set_Y,0.1)
	RRW1 = getRRW(std_training_set_X,training_set_Y,1)
	print "-------------------Linear and ridge regression-------------------"	
	print "Linear Regression Train set MSE = ",getMSE(std_training_set_X,training_set_Y,W)
	print "Ridge Regression Train set MSE (lambda=0.01) = ",getMSE(std_training_set_X,training_set_Y,RRW001)
	print "Ridge Regression Train set MSE (lambda=0.1) = ",getMSE(std_training_set_X,training_set_Y,RRW01)
	print "Ridge Regression Train set MSE (lambda=1) = ",getMSE(std_training_set_X,training_set_Y,RRW1)

	print "Linear RegressionTest set MSE = ",getMSE(std_testing_set_X,testing_set_Y,W)
	print "Ridge Regression Test set MSE (lambda=0.01) = ",getMSE(std_testing_set_X,testing_set_Y,RRW001)
	print "Ridge Regression Test set MSE (lambda=0.1) = ",getMSE(std_testing_set_X,testing_set_Y,RRW01)
	print "Ridge Regression Test set MSE (lambda=1) = ",getMSE(std_testing_set_X,testing_set_Y,RRW1)
	
	best_lam = crossValidation(std_training_set_X,training_set_Y,std_testing_set_X,testing_set_Y)


	
	#best 4
	print"----------------Get 4 best features directly from correlation----------------"
	best_4_features = np.array(abs_correlation).argsort()[-4:][::-1]
	print "best 4 features by correlation",[feature_names[i] for i in best_4_features]
	std_training_set_X_4_feature = std_training_set_X[:,np.append(np.array(0),best_4_features)]
	std_testing_set_X_4_feature = std_testing_set_X[:,np.append(np.array(0),best_4_features)]
	best_4_W = getW(std_training_set_X_4_feature,training_set_Y)
	print "MSE for training set ", getMSE(std_training_set_X_4_feature,training_set_Y,best_4_W)
	print "MSE for testing set ", getMSE(std_testing_set_X_4_feature,testing_set_Y,best_4_W)

	
	
	#iterative best 4
	print"--------------Get 4 best features iteratively from correlation--------------"
	std_training_set_X_iterative_current = np.array(std_training_set_X)
	residue = np.array(training_set_Y)
	selected_features = np.array([],dtype='int')
	for i in range(1,5):
		abs_current_correlation = map(abs,getCorrelation(std_training_set_X_iterative_current,residue))
		#print "abs",abs_current_correlation
		best_feature = np.array(abs_current_correlation).argsort()[-1:][::-1][0]
		selected_features = np.append(selected_features,best_feature)
		#print "iteration:",i,":selected features:",selected_features
		std_training_set_X_iterative_current[:,best_feature] = 0.01
		X_used = std_training_set_X[:,np.append(np.array(0),selected_features)]
		iterative_W = getW(X_used,residue)
		residue = np.array(training_set_Y - np.dot(X_used,iterative_W))
	std_train_X_iterative = std_training_set_X[:,np.append(np.array(0),selected_features)]
	std_test_X_iterative = std_testing_set_X[:,np.append(np.array(0),selected_features)]
	iterative_W = getW(std_train_X_iterative,training_set_Y)
	print "selected features:",[feature_names[i] for i in selected_features]
	print "MSE for training set ", getMSE(std_train_X_iterative,training_set_Y,iterative_W)
	print "MSE for testing set ", getMSE(std_test_X_iterative,testing_set_Y,iterative_W)
	
	#brute force
	print"-------------------Brute force-------------------"
	brute_force_list = getBruteForceList(0,std_training_set_X,training_set_Y,np.array([],dtype='int'),[])
	#print brute_force_list
	bestMSE = 9999999999999
	bestList = []
	#for i in range(len(brute_force_list)-1,0,-1):
	for i in range(0,len(brute_force_list)):		
		std_training_set_brute = std_training_set_X[:,np.append(np.array(0),brute_force_list[i])]
		brute_W = getW(std_training_set_brute,training_set_Y)
		brute_MSE = getMSE(std_training_set_brute,training_set_Y,brute_W)
		if brute_MSE < bestMSE:
			bestMSE = brute_MSE
			bestList =  brute_force_list[i]
	print "Best features by brute force are:",[feature_names[i] for i in bestList]
	std_training_set_brute = std_training_set_X[:,np.append(np.array(0),bestList)]
	brute_W = getW(std_training_set_brute,training_set_Y)
	print "MSE for training set ", getMSE(std_training_set_brute,training_set_Y,brute_W)
	std_testing_set_brute = std_testing_set_X[:,np.append(np.array(0),bestList)]
	print "MSE for testing set ", getMSE(std_testing_set_brute,testing_set_Y,brute_W)

	

	
	#Polynomial Feature Expansion
	print"-------------------Polynomial Feature Expansion-------------------"
	training_set_X_poly = np.array(training_set_X)
	testing_set_X_poly = np.array(testing_set_X)
	for i in range(1,14):
		x = np.multiply(np.array(np.array(training_set_X_poly)[:,i]),np.array(np.array(training_set_X_poly)[:,i]))
		x = x.reshape(433,1)
		training_set_X_poly = np.concatenate((training_set_X_poly,x),axis=1)
		for j in range(i+1,14):
			if i != j:
				x = np.multiply(np.array(np.array(training_set_X_poly)[:,i]),np.array(np.array(training_set_X_poly)[:,j]))
				x = x.reshape(433,1)
				training_set_X_poly = np.concatenate((training_set_X_poly,x),axis=1)
		
		x = np.multiply(np.array(np.array(testing_set_X_poly)[:,i]),np.array(np.array(testing_set_X_poly)[:,i]))
		x = x.reshape(73,1)
		testing_set_X_poly = np.concatenate((testing_set_X_poly,x),axis=1)
		for j in range(i+1,14):
			if i != j:
				x = np.multiply(np.array(np.array(testing_set_X_poly)[:,i]),np.array(np.array(testing_set_X_poly)[:,j]))
				x = x.reshape(73,1)
				testing_set_X_poly = np.concatenate((testing_set_X_poly,x),axis=1)
			
	poly_mu,poly_sigma = getStandardizeParams(list(training_set_X_poly))


	std_training_set_X_poly = np.array(standardizeSet(training_set_X_poly,poly_mu,poly_sigma))
	std_testing_set_X_poly = np.array(standardizeSet(testing_set_X_poly,poly_mu,poly_sigma))
	poly_W = getW(std_training_set_X_poly,training_set_Y)
	print "MSE for training set: ",getMSE(std_training_set_X_poly,training_set_Y,poly_W)
	print "MSE for testing set: ",getMSE(std_testing_set_X_poly,testing_set_Y,poly_W)
	




main()

