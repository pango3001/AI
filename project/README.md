# Semester-long course project

The [activities](./activities/README.md) that we will cover, will help you learn all the background to complete this semester-long course project.

## A. Objectives
The overall goal of working on this semester long project is to learn the foundations of using Tensorflow/Keras to build, train, and evaluate feed-forward neural networks on standard tabular data that can be framed as a classification or a regression problem. If you are learning machine learning for the first time, a binary classification problem will probably be easier (not a regression problem). A problem is 'binary classification' if your output column is 0 or 1.  If you are predicting continous values using a pre-cleaned tabular dataset. While working on the project you will compare the performance (accuracy, speed, etc.) of various  neural network models including a basic single layer model. For a regression problem, the single layer model is a linear regression model, and for a binary classification problem, the single layer model is a logistic regression model. You will also learn to investigate "what", "how", and "why" a model makes predictions. Following are the restrictions when choosing a dataset:
1. NO time-series data (for example stock market prediction), NO image data,  NO text data (natural language processing)
1. Data must have at least a 1000 rows and at least 3 input features (columns) 

## B. Expectations
1. You will work on your projects individually (i.e. group submissions are not allowed).
1. Reports for all phases (including the final report) must be prepared using <a href="https://www.overleaf.com/">Overleaf</a>. Non-overleaf submissions will receive a 0 (zero). You are free to use any templates you want. [Here](https://www.overleaf.com/read/vgckqpfdyrwp) is an example. You can learn more about Overleaf [here](https://www.overleaf.com/learn/latex/LaTeX_video_tutorial_for_beginners_(video_1)). If you have accessibility needs please email me and I will waive this requirement.

## C. Phases
**In each phase you are exepected to submit:**  
1. An HTML version of the notebook
   - If you are using Google Colab, please convert the notebook to `.html` files and submit the `.html` files, for example using [htmltopdf](https://htmtopdf.herokuapp.com/ipynbviewer/).
1. A PDF report describing your findings (downloaded from your Overleaf project). The reports for the first three phases can be as long as you make them but the final report has limit on the number of pages. 
1. A link to view your Overleaf project.

Below is the list of all phases and the outline of what you will be working on in each phase. 

### Phase I. Data analysis & preparation
1. Watch [how to clean a tabular dataset for machine learning](https://youtu.be/0bj6KbEUJ_o).
1. Discuss why you chose to work on this project.
1. Describe the dataset and its source.
1. Visualize/plot the distributions of each input features and discuss the range of the values (min, max, mean, median, etc.). For example, plot histograms showing distribution of each input features.
1. Discuss the distribution of the output labels. In the case of classification, check if the data is imbalanced by calculating what percentage of the output labels are 0 and what percentage are 1. If your dataset is heavily imbalanced (for example, 1% vs 99%) it may be easier if you choose a different dataset. In the case of regression, check if the values are uniformly distributed or not by plotting the distribution of the output variable.
1. Discuss how you normalized your data.

[Here](https://github.com/zegster/artificial-intelligence/blob/master/data_analysis_and_preparation/Data_Analysis_and_Preparation.pdf) is an example report.

### Phase II. Model selection & evaluation
1. Split your data into training, and validation sets
1. Compare the results of the neural network with a linear regression or logistic regression model
    - Start with a basic model and then grow your model into a multi-layered model
    - Discuss how neural network models will be selected
    - Document your performance comparison
1. Study performance difference when linear activation is used instead of sigmoid (and vice versa)
   - How does your performance change when linear activations are used instead of sigmoid, in the last neuron and all other neurons?
     ```python
     # Change 'relu' to 'linear' and 'sigmoid' in the layers below
     model = Sequential()
     model.add(Dense(9999, input_dim = len(X[0, :]), activation='relu'))
     ...
     model.add(Dense(9999, activation='relu'))
     ```
1. Plot your learning curves and include them in your report
1. Discuss what architecture (how big) you need to overfit the data
1. Discuss what architecture (how big) you do need to overfit when you have output as additional input feature
1. Evaluate your predictions (using Precision, Recall, MAE, MSE, etc.)
1. [OPTIONAL] Code a function that represents your model
   - After your model is trained, read all the weights, and build your own function/method that serves as the model
   - Verify that predictions you obtain are same as the one you obtained using your trained model

[Here](https://github.com/zegster/artificial-intelligence/blob/master/model_selection_and_evaluation/Model_Selection_Evaluation.pdf) is an example report.

### Phase III. Feature importance and reduction
1. Study feature importance by iteratively removing input features
1. Identify non-informative input features and remove them
1. Compare your feature-reduced model with the original model with all input features

[Here](https://github.com/SoderstromJohnR/CS4300Final/blob/master/Phase%203%20Report.pdf) is an example report.

### Phase IV. Report
**What to submit?**   
1. A copy of your final report    
    * Your report must not be very long; 10/12 pages at most.
    * All tables and figures must be numbered and captioned/labelled.
    * Don't fill an entire page with a picture or have pictures hanging outside of the page borders.
    * It is encouraged but not required to you host your project (and report) at Github.  
    * Turn off the dark mode in Notebook before you copy images/plots (the lables and ticks are hard to see in dark mode).
    * Your report should include abstract and conclusion (each 250 words minimum).
1. A link to your final Notebook

## D. Sample final reports (from previous semesters)
* EPL game result prediction by Bikash Shrestha - [report](https://github.com/badriadhikari/AI-2020spring/blob/master/sample-reports/sample-report-1.pdf)
* Prediction of housing prices by Syeda Afrah Shamail - [report](https://github.com/afrah1994/Prediction-of-Housing-Prices/blob/master/Final%20report.pdf)
* Factors in tennis by John Soderstrom - [report](https://github.com/SoderstromJohnR/CS4300Final/blob/master/Final%20Report.pdf)
* Predicting pulsar stars by Duc Ngo - [report](https://github.com/zegster/artificial-intelligence/blob/master/final_assembly/Final_Assembly.pdf)

