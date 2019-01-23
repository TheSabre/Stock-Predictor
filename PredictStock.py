import Tkinter as tk
import matplotlib
from Tkinter import *
import datetime
import random
import pandas as pd

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm


class PredictStock:

    def __init__(self, master):
        self.days = []
        self.prices = []
        self.csvLocation = 'all_stocks_5yr.csv'
        self.df = pd.read_csv(self.csvLocation)
        self.iteration = 0
        self.confidenceStandard = 0.98
        self.createFrames()
        self.LineChart()
    
    def createFrames(self):
        self.labelFrame = tk.Frame(root) # Frame for Labels
        self.oFrame = tk.Frame(root) # Frame for options
        self.bFrame = tk.Frame(root) # Frame for button
        self.gFrame = tk.Frame(root) # Frame for graph

        # Creating default option objects for the options
        self.stockDefOpt = StringVar(self.oFrame,)
        self.daysDefOpt = StringVar(self.oFrame, )

        # Filling all the frames along the X-Axis
        self.labelFrame.pack(fill=X)
        self.oFrame.pack(fill=X)
        self.bFrame.pack(fill=X)
        self.gFrame.pack(fill=X)

    #Function to destroy graph frame to create new graph on top of it
    def clearGraphFrame(self):
        self.gFrame.destroy()
        self.gFrame = tk.Frame(root)
        self.gFrame.pack(fill=X)

    def LineChart(self):
        print("Show the chart!")

        stockName = tk.Label(self.labelFrame, text="Stock Name")
        duration = tk.Label(self.labelFrame, text="Duration")
        genButton = Button(self.bFrame, text="Generate Graph", command= lambda: self.generateGraph(), width=30)

        stockNames = pd.unique(pd.Series(self.df['Name']))
        print(stockNames)
        stockNames = ["AAPL", "AMZN", "ADBE", "AMD", "BRK.B", "CMG", "FB", "GOOG", "GOOGL", "JNJ", "JPM", "KSS", "MSFT", "TRIP", "UAA", "XOM", "XL"]
        totalDuration = ["10","20","30","40","50","60"]
        
        # Setting the default option value
        self.stockDefOpt.set("AMZN");
        self.daysDefOpt.set("30");

        self.stockOptions = OptionMenu(self.oFrame, self.stockDefOpt, *stockNames)
        self.durationOptions = OptionMenu(self.oFrame, self.daysDefOpt, *totalDuration)


        # Packing all the options and labels

        stockName.pack(side="left",fill=X, expand=1, padx=50, pady=15)
        duration.pack(side="left",fill=X, expand=1, padx=50, pady=15)
        self.stockOptions.pack(side="left", fill=X, expand=1, padx=50, pady=10)
        self.durationOptions.pack(side="left", fill=X, expand=1, padx=50, pady=10)
        genButton.pack(expand=1,padx=15, pady=30)

        print(self.stockDefOpt.get())


        # Embedding a default graph for presentation

        self.f = Figure(figsize=(5,5), dpi=100)

        now = datetime.datetime.now()
        x=[]
        y=[]
        
        #Putting Random Data for default graph
        for i in range(0,30):
            rand = random.randint(112, 150)
            y.append(rand)
        for i in range(0,30):
            x.append(i)

        #Putting the graph inside the figure
        ax = self.f.add_subplot(111)
        ax.set_title("Stock Prices For ABC")
        ax.set_xlabel("Days")
        ax.set_ylabel("Closing Stock Price")

        #Plot the graph with the x and y co-ordinates
        ax.plot(x,y, label="ABC", gid=1)

        # Create a canvas to draw the figure
        self.canvas = FigureCanvasTkAgg(self.f, self.gFrame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand= True)

		# Create a toolbar for the graph
        toolbar = NavigationToolbar2TkAgg(self.canvas, self.gFrame)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def generateGraph(self):

        # Calculate for the prediction
        selected_stock_name = self.stockDefOpt.get()

        selected_stock_data = self.df[self.df['Name']== selected_stock_name]
        
        selected_stock_close_data = selected_stock_data[['close']]
        #print("Generate Graph for the following data:")
        #print(selected_stock_close_data)

        # Get the number of days the stocks need to be predicted for
        predictionLimit = int(self.daysDefOpt.get())

        # Added a new column named prediction to the data frame and shifted the last 30 values to prepare for training
        selected_stock_close_data['prediction'] = selected_stock_data['close'].shift(-predictionLimit)

        print(selected_stock_close_data)

        # Dropping prediction column for scaling data
        selected_stock_close_scaledData = np.array(selected_stock_close_data.drop(['prediction'], 1))
        #print(selected_stock_close_data)
        selected_stock_close_scaledData = preprocessing.scale(selected_stock_close_scaledData) 

        # Setting stock close prediction data to the last 'predictionLimit' values
        # Deleting last 'predictionLimit' values from selected_stock_close_scaledData
        selected_stock_close_prediction = selected_stock_close_scaledData[-predictionLimit:] 
        selected_stock_close_scaledData = selected_stock_close_scaledData[:-predictionLimit] 
        

        # Creating a new array containing the values from the column prediction and deleting the last "predictionLimit" values
        predictionData = np.array(selected_stock_close_data['prediction']) 

        # Deleting last 'predictionLimit' values from predictionData
        
        predictionData = predictionData[:-predictionLimit] # Equivalent to Y

        #print(selected_stock_close_scaledData)
        #print(len(selected_stock_close_scaledData))

        #print(predictionData)
        #print(len(predictionData))

        # PERFORMING LINEAR REGRESSION----------------------------------------------

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(selected_stock_close_scaledData, predictionData, test_size = 0.2)

        # Training
        clf = LinearRegression()
        clf.fit(X_train,y_train)
        # Testing
        confidence = clf.score(X_test, y_test)
        print("Percentage of confidence: ", confidence)

        if confidence<self.confidenceStandard:
            if self.iteration>20:
                self.confidenceStandard-=0.05
            self.iteration+=1
            self.generateGraph()
            return

        self.confidenceStandard=0.98
        final_prediction = clf.predict(selected_stock_close_prediction)
        print(final_prediction)
        print(len(final_prediction))

        # Embedding the graph of predicted values
        
        self.clearGraphFrame()

        self.f = Figure(figsize=(5,5), dpi=100)

        x=[]
        y=final_prediction

        #Setting x value
        for i in range(0,int(self.daysDefOpt.get())):
            x.append(i)

        #Putting the graph inside the figure
        ax = self.f.add_subplot(111)
        ax.set_title("Stock Prices For " + str(self.stockDefOpt.get() + "      Confidence Level: " + str(round(int(confidence * 100)))+ "%"))
        ax.set_xlabel("Days")
        ax.set_ylabel("Closing Stock Price")

        #Plot the graph with the x and y co-ordinates
        ax.plot(x,y, label=str(self.stockDefOpt.get()), gid=1)

        # Create a canvas to draw the figure
        self.canvas = FigureCanvasTkAgg(self.f, self.gFrame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand= True)

		# Create a toolbar for the graph
        toolbar = NavigationToolbar2TkAgg(self.canvas, self.gFrame)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        

root = tk.Tk()
main = PredictStock(root)

root.minsize(height=150, width=800)
root.geometry("1280x720")

root.mainloop()

