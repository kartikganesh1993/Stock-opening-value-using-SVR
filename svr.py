#usr/bin/python!

#PROJECT NAME : PREDICTING STOCK OPENING VALUE FROM PREVIOUS DATA.
##GROUP MEMBERS : Bopin Valsan, Kartik Ganesh, Harshvardhan Singh, Aayush Vats, Gaurav Bhatt, Jitender Joshi
###EMAIL ID : pgp17bopinvalsan@imt.ac.in,pgp17kartiknagrajan@imt.ac.in,pgp17harshvardhansingh@imt.ac.in,pgp17aayushvats@imt.ac.in,pgp17gauravbhatt@imt.ac.in,pgp17jitenderjoshi@imt.ac.in
#### REGISTRATION NUMBERS : 170101023, 170103097, 170101031, 170103005, 170103071, 170103094
##### CONTACT NUMBERS : +91 9036015558 , +91 9003961463 , +91 9999397672, +91 9880977119 , +91 8879246391 , +91 8860191543
###### PYTHON VERSION : python 2 and 3 
####### VERSION : 0.1.1
######## DATE CREATED : 20/08/2018

######### FUNCTIONS EXPLAINED with Working : get_data() basically will store the data in the data set into arrays of prices and dates separately. This will further be used to compute the predicted price . The dates are separated by the use of the split function where we search for a - in the dates.. The input here is simply that of the filename that contains the required data.
########## : the predict_data() is majorly the crux and is the part with the SVR. we have gone for testing with the kernels linear , RBF and polynomial and have plottedd out the respective outputs and determined based on the graphs which model fits the data best and used that predicted value. The inputs for this function are the dates array created as a global vriable and filled in the function get_data() along with prices which is the same process as dates. The third input is the date value for which the prediction is to be made.


########### ASSUMPTIONS AND FAILURE CASES : Since this is a very crude and basic approach in this project the focus is to avoid external factors and market impacts that happen such as mergers, acquisitions , new announcements and other external factors that as of now cannot be accounted for or cant be included. 
#It also assumes that the data is available for each day as we do not expect values for stocks to be not available for a day irrespective of the volume sold or bought that day.
#The model also assumes that the data is real and follows the unstructured and unorganized pattern a stock market does or there would be better methods to do the same.
#Weekend model assumption is not taken into account and as of now the value will be shown for the opening value on Sat and Sun as well.
#All the prices have the same unit i.e all are in dollars or rupees and so on without change across the data.
#Date format is as dd-mm-yy with the – and not / or some other value in words.

import csv   # to read the dataset of csv format here.
import numpy as np # numpy to work with multidimensional arrays
from sklearn.svm import SVR # the main class from which SVR function is used for the regression 
import matplotlib # used primarily to get the plot calling the subsequent pyplot
matplotlib.use('Agg') # remove DISPLAY error tkinter or tclerror that comes up 
import matplotlib.pyplot as plt

dates = [] #array empty initialised for storing dates from the dataset
prices = [] #empty array initialised to store the price from the dataset

###  This function will be retrieving the data from the file and ensuring that the dataset can be created to have a prediction made.
def get_data(filename):   
	with open(filename, 'r') as csvfile:  # opening the CSV file
		csvFileReader = csv.reader(csvfile)  #reading the CSV file
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0])) #adding values found in dates by splitting based on dates as -  to the dates array initialised before
			prices.append(float(row[1])) #adding values found in dataset row 1 to price array initialised before
	return

def predict_price(dates, prices, x): # created a function that will use the SVR to predict the opening price of the next day.
	dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1

	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models with the SVR function and specifying the kernels to be used. By default the value taken up would be the RBF model.
	svr_lin = SVR(kernel= 'linear', C= 1e3) # C here mentions the error allowed on either side.
	svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
	svr_rbf.fit(dates, prices) # fitting the data points in the models
	svr_lin.fit(dates, prices)
	svr_poly.fit(dates, prices) #basically using the variables created as objects and passing arguments to the function fit with dates and prices.

	plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
        plt.savefig('RBF model.png') #Saving the RBF model graph
        plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
        plt.savefig('Linear_model.png') #Saving the Linear Model graph
        plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
        plt.savefig('Polynomial_model.png') #Saving the Polynomial Model graph
        plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()
        plt.savefig('Support Vector Regression.png') #combined graph to show all three models against the data
	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('goog.csv') # calling get_data method by passing the csv file to it
print "Dates- ", dates #print the dates that the get_data function retrieved from the dataset which are stored in the variable dates
print "Prices- ", prices #print the prices that the get_data function retrieved from the dataset which are stored in the variable prices

predicted_price = predict_price(dates, prices, 26)  # passing arguments to the predict_price function
print "\nThe stock open price for next day of GOOGLE is:"
print("For RBF kernel: $", (str(predicted_price[0]))) #Printing the output for RBF kernel
print( "For Linear kernel: $", str(predicted_price[1])) # printing the output for linear kernel
print( "For Polynomial kernel: $", str(predicted_price[2])) #printing the output for polynomial kernel
	 
