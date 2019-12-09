# LSTM-StockPrediction-

Stock Prediction Using Reactionary Recurrent Neural Network

Results:

normal training loss one: 0.00872524 stacked training loss: 0.0051617934

normal testing loss: 0.0065551633 stacked testing loss: 0.0006514265

Contributors: Mustafa Ghani (mghani1), Rahul Dey (rdey2)

Introduction
 
1. What problems are you trying to solve and why?
 
Recurrent Neural Networks are inherently deep on a temporal scale. However, unlike conventional DNNs they do not have a hierarchy of layers. Our goal is to experiment with the effect that spatial depth has on the predictive power of RNNs. To this end, we plan to implement a stacked LSTM architecture to predict future stock prices. The first layer will use available time-series financial data to learn an RNN model that predicts stock price. The second layer will use the output sequence of the first as a new feature-set to make a derivative prediction. Our goal is to allow our model the flexibility to learn a predictive strategy in response to our initial layer, mimicking an environment of competing algorithms.
 
 2. If you are implementing an existing paper, describe the paper’s objectives and why you chose this paper/ If you are doing something new, detail how you arrived at this topic and what motivated you.
 
We are not implementing an existing paper, but we have taken inspiration from a paper by Michiel Hermans and Benjamin Schrauwen of Ghent University, in which they study the effect of a hierarchy of recurrent neural networks on processing time series data. They mention how RNN’s falls short of Deep Neural Networks as they lack hierarchical processing. Their paper thus combines DNNs with RNNs as “each layer in the hierarchy is a recurrent neural network, and each subsequent layer receives the hidden state of the previous layer as input time series” (Herman et al). Therefore, to create such a temporal hierarchy we plan to apply a similar logic, stacking LSTMs and using the initial output sequence as input for the subsequent layer. Most implementations of DNNs for stock prediction involve only a single layer LSTM or GRU. We thought it would be interesting to test how improved accuracy with Deep LSTMs may apply to the stock market.   
 
3. What kind of problem is this? Classification? Regression? Structured prediction? Reinforcement Learning? Unsupervised Learning? Etc.
 
This is a regression problem. Theoretically we assume that the neural  network is learning a function that correlates the behavior of select attributes with the movement of stock price. 
 
Related Work: Are you aware of any, or is there any prior work that you drew on to do your project?

We will be using a Kaggle  implementation of LSTM for stock prediction as a reference outline.

Data
 
Where is your data coming from?
 
We are using company fundamentals and financial data (time series) extracted from the Yahoo Finance API. Our data comprises the opening and closing share prices of 1782 publicly traded companies on NASDAQ.
 
How big is it? 
 
We have 7 years (2010-2016) of price data for each stock on the NASDAQ. There are 1782 stocks and each have attributes of opening price, closing price, high, low and volume.
 
Will you need to do significant preprocessing?
 
We will first need to parse the dataset for attributes ( high, low, opening price, closing price and volume) corresponding to a single stock for the entire series of timestamps and repeat this process. Then group all the examples by industry and finally normalize all the price metrics.
 
Methodology
 
What is the architecture of your model?
 
We are using a stacked LSTM architecture as shown below.
 
How are you training the model?
 
Since we have 7 years of data, we will be training the model on the first 6 years of data and then validating and testing on year 7. However, we could vary the length of the training sample and use incremental batches (i.e. 2 years to predict the third and so forth). Furthermore we will be using a window size of 2 months for training. 
 
If you are implementing an existing paper, detail what you think will be the hardest part about implementing the model here.
 
The paper that we are drawing inspiration from uses the hidden state output of the layer as input for the second layer. On the other hand, we plan to use the output sequence as input for the second LSTM. Figuring out how to align this new time series data with our old labels in a manner that yields improved prediction will be tricky. 
 
Metrics
 
What constitutes “success?”
 
Success will be determined by a superior accuracy for the stacked LSTM architecture in comparison to the single-LSTM model. We will be calculating loss by taking the mean-squared error across all predictions. 
 
What experiments do you plan to run?
 
We plan to vary the degree of overfitting allowed for the first layer to see the effect that it has on the accuracy of the stacked layers. 
 
For assignments, we have looked at the accuracy of the model. Does the notion of accuracy apply for your project?
 
Accuracy is applicable as we can directly measure the difference between predicted price and actual price.
 
If you are implementing an existing project, detail what the authors of that paper were hoping to find and how they quantified the results of their model.
 
While we are not replicating the paper, Herman et al were attempting to  measure the effect of a hierarchy on RNNs ability to process time series data for the prediction of next characters in Wikipedia’s corpus of text. Therefore, they used the normalized average distance between “hidden states of a perturbed and unperturbed network as a function of presented characters” to quantify the results of their model.
 
If you are doing something new, explain how you will assess your model’s performance.
 
We will be comparing the accuracy for the sing-layer LSTM to that of the stacked LSTM.
	
Ethics
 
What is your dataset? Are there any concerns about how it was collected, or labeled? Is it representative? What kind of underlying historical or societal biases might it contain?
 
Our data set is a collection of stock prices of publicly traded companies in the NYSE. This data represents the opening and closing prices of 1782 companies collected across 6 years (from 2010 to 2016). The data set represents the market’s sentiments towards the companies across a period of time and may be skewed towards certain companies that have a strong brand/household name. We attained it from Yahoo Finance’s API which is very reliable and well-known.
 
Who are the major “stakeholders” in this problem, and what are the consequences of mistakes made by your algorithm?
 
The major stakeholders in this problem would be investors that use predictions based on our algorithm to invest in stock with the aim of maximising returns. Consequences of mistakes in the algorithm would directly result in investors losing their money. 
 
Another effect a mistake in the algorithm could have is misleading the market. With a large enough fund, targeted investing in one company would drive up stock prices skewing market views and overvaluing a company. 
 
How are you planning to quantify or measure error or success? What implications does your quantification have?
 
We plan to calculate a mean squared error based on the difference between predicted prices and the actual prices. However this may not account for the possibility that despite there being an offset from the actual price, the predicted prices may be correlated or follow a similar trend to that of the actual price. This is because MSE only measures how close a regression is to a sequence of points. 
 
Division of labor: Briefly outline who will be responsible for which part(s) of the project.
This project is undertaken by two people (Mustafa and Rahul) and we  plan to split the work equally among ourselves. Mustafa will be responsible for the initial preprocessing and implementation of the first LSTM layer. Rahul will be working on initializing the hyperparameters to optimize outputs of the first LSTM layer and then will implement the second layer. The input for the second layer( output from first LSTM) will be preprocessed by Mustafa. Equal effort is expected to be put in training the model, improving accuracy and debugging. 

