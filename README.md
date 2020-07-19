# Stockastic

Gone are the days when you had to train stock forecasting models in a convoluted Python notebook filled with subpar helper functions and comments that make you want to claw your eyes out.


![Proof of Concept](proof_of_concept.gif)


## Table of Contents

* [Introduction](#introduction)
  * [Stockastic](#stockastic)
  * [Use Cases](#use-cases)
* [Built With](#built-with)
* [How It Works](#how-it-works)
  * [Models Used](#models-used)
  * [Data Flow](#data-flow)
* [Scope of Improvement](#scope-of-improvement)
* [Acknowledgements](#acknowledgements)


## Introduction

### Stockastic
Stockastic is a functionally abstracted, end-to-end data science application that lets you visualize, model, and forecast any S&P 500 stock without writing a single line of code.
  
### Use Cases
Some use cases of this stock analysis application are 
  <li>Beginners who are trying to understand the correlation between different parameters in a time series machine learning model and their impact on the model output</li>
  <li>Time series modelling ninjas who are tired of looking at their dry Python notebooks with dozens and dozens of cells of poorly written code and are looking for a pleasant user-facing modelling experience</li>
   

### Built With
* [Python](https://www.python.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [Plotly](https://plotly.com/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
* [Semantic UI](https://semantic-ui.com/)
* [My Only Two Brain Cells](https://www.bbc.com/news/uk-england-sussex-36443264#:~:text=Snails%20use%20two%20brain%20cells,it%20if%20food%20was%20present.)


## How It Works
The application makes use of 5 predictive models that work on time series data (such as historic stock prices) to generate predictions. It lets the user view the historic price of the stock, build a model by tuning the model parameters, and instantly visualize the results with Plotly graphs. The user can then experimentally forecast with the generated model.
  
### Models Used
<a href="https://otexts.com/fpp2/moving-averages.html"><li>Moving Average</li></a>
  The model parameters that are currently tunable for the moving average model are
  * The number of days considered in the <a href="https://www.mathworks.com/matlabcentral/answers/315739-how-to-decide-window-size-for-a-moving-average-filter">window</a> of the moving average
  * The <a href="https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6#:~:text=both%20of%20them!-,Train%2FTest%20Split,to%20other%20data%20later%20on.&text=Pandas%20%E2%80%94%20to%20load%20the%20data,frame%20and%20analyze%20the%20data.">train-test split ratio</a> aka the number of data points that are used to train the model and test the model respectively
<br>

<a href="https://otexts.com/fpp2/regression.html"><li>Linear Regression</li></a>
  The model parameters that are currently tunable for the linear model are
  * The <a href="https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6#:~:text=both%20of%20them!-,Train%2FTest%20Split,to%20other%20data%20later%20on.&text=Pandas%20%E2%80%94%20to%20load%20the%20data,frame%20and%20analyze%20the%20data.">train-test split ratio</a> aka the number of data points that are used to train the model and test the model respectively
<br>

<a href="https://www.researchgate.net/publication/321206629_A_methodology_for_applying_k-nearest_neighbor_to_time_series_forecasting"><li>K-Nearest Neighbors (KNN)</li></a>
  The model parameters that are currently tunable for the KNN model are
  * The <a href="https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6#:~:text=both%20of%20them!-,Train%2FTest%20Split,to%20other%20data%20later%20on.&text=Pandas%20%E2%80%94%20to%20load%20the%20data,frame%20and%20analyze%20the%20data.">train-test split ratio</a> aka the number of data points that are used to train the model and test the model respectively
  * The number of neighbors to use for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors">kneighbors</a> queries
  * The <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">weight function</a> you can apply to the neighbors
  * The <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">power parameter</a> (p-value) for the Minkowski distance metric
<br>

<a href="https://www.researchgate.net/publication/321206629_A_methodology_for_applying_k-nearest_neighbor_to_time_series_forecasting"><li>Long Short Term Memory (LSTM)</li></a>
  The model parameters that are currently tunable for the LSTM model are
  * The <a href="https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6#:~:text=both%20of%20them!-,Train%2FTest%20Split,to%20other%20data%20later%20on.&text=Pandas%20%E2%80%94%20to%20load%20the%20data,frame%20and%20analyze%20the%20data.">train-test split ratio</a> aka the number of data points that are used to train the model and test the model respectively
  * The <a href="https://stackoverflow.com/questions/59995733/effect-of-number-of-nodes-in-lstm">number of nodes</a> in each layer of the neural network
  * The number of <a href="https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9">epochs</a>
<br>

<a href="https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/"><li>Autoregressive Integrated Moving Average (Auto-ARIMA)</li></a>
  The model parameters that are currently tunable for the Auto-ARIMA model are
  * The <a href="https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6#:~:text=both%20of%20them!-,Train%2FTest%20Split,to%20other%20data%20later%20on.&text=Pandas%20%E2%80%94%20to%20load%20the%20data,frame%20and%20analyze%20the%20data.">train-test split ratio</a> aka the number of data points that are used to train the model and test the model respectively</li>
  * The number of past values used for forecasting (<a href="https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/">p</a>)</li>
  * The forecasting error considered to further predict values (<a href="https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/">p</a>)
  * The order of first-differencing (<a href="https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/">d</a>)
  * The order of seasonal-differencing (<a href="https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/">D</a>)

### Data Flow
<ol>
  <li>
    <h4>Data Collection</h4>
   The initial dataset containing the historic stock price is obtained from <a href="https://ca.finance.yahoo.com/">Yahoo Finance</a>. 

   To obtain the csv file containing the historic stock performance from Yahoo Finance (past 5y performance, present day - 5y ago), we have to start by calculating the <a href="https://www.unixtimestamp.com/">Unix timestamp</a> of the present date and the date 5y ago. We can then proceed to building the URL string to read the stock data into a pandas DataFrame.
   
   ```python
    # Calculates Unix timestamps for start and end dates to fetch data
    timestamp_today = int(datetime(int(dtc[0]), int(dtc[1]), int(
        dtc[2]), 0, 0).replace(tzinfo=timezone.utc).timestamp())
    timestamp_five_years_ago = int((datetime(int(dfyc[0]), int(dfyc[1]), int(
        dfyc[2]), 0, 0)).replace(tzinfo=timezone.utc).timestamp())
        
    # Reading in stock data from Yahoo Finance in the above timestamps' range
    csv_url = "https://query1.finance.yahoo.com/v7/finance/download/" + ticker + \
        "?period1=" + str(timestamp_five_years_ago) + "&period2=" + \
        str(timestamp_today) + "&interval=1d&events=history"
    df = pd.read_csv(csv_url)
   
   ```
 </li>
 
 <li> 
  <h4>Data Processing</h4>
 This step involves allowing the user to tune the model hyperparameters, submit them through a form, and then retrieve the hyperparameter values on the backend so a model can be built with those values.
 
 ```
 Hyperparameters value submission through HTML form
 
 <form action="model_name/customize" method="POST">
   <input name="hyperparameter name" />
   .
   .
   .
   <input type="submit">
 </form>
 ```
 
 ```python
 # Retrieving hyperparameters values through Flask route
 @model_name.route("model_name/customize", methods=["POST"])
 def model_name_customize_input():
   ticker = request.form["ticker"]
   parameter_name = request.form["parameter_name"]
   .
   .
   .
   
   return redirect("/" + ticker + "/model_name/customize/" + parameters)
 ```
 </li>
 
 <li>
 <h4>Data Visualization</h4>
 This is done exclusively through Plotly's baseline figures to which traces of the testing, validation, and forecast data points are added. All of the visualizations have been dimensioned to a default width to height ratio of 2.5 (1000px : 400px). 
 
 ```python
 # Graph by starting with a baseline figure and adding traces
 fig_model_name = go.Figure(layout=fig_layout)
 
 fig_model_name.add_trace(
 	x="date_here",
    y="price_here",
    mode="lines",
    name="Testing/Validation/Forecast"
 )
 
 # Setting figure layout
 fig_layout = go.Layout(
    autosize=False,
    width=1000,
    height=400
)
```

</li>

<li>
<h4>Forecasting</h4>
The forecasting process for all the models follows these general steps <br>
• Obtain the date (year, month, day) that the user wants to predict the price on through a form submission process similar to the one discussed in "Data Processing" <br>
• Generated a pandas DataFrame containing dates from present day to the date of forecast

```python
def generate_dates_until(year, month, day):
    """
    Generates dates between today and the date determined by the params
    and returns them in a pandas DataFrame
    """

    start_date = date.today()
    end_date = date(year, month, day)
    dates_dict = {"Date": pd.date_range(start=start_date, end=end_date)}
    dates_df = pd.DataFrame(data=dates_dict)

    return dates_df
```
• Pass the DataFrame containing all the dates into the model that is being used to serve the forecast
</li>
</ol>


## Scope of Improvement
There's certainly a ton of changes that can be implemented to better the application. I plan to attempt implementing/further exploring some of these changes over the new few months (roughly in this order)
<li>The current runtime of the LSTM and Auto-ARIMA models are about 45s and 3m respectively. This isn't ideal as I envisioned the feedback for this application being almost instantaneous - Will probably look into running these models using cloud TPUs</li>
<li>Quality and quantity of customizable hyperparameters - This can certainly be improved by feature engineering, exploring deeper built-in hyperparameters etc. (Currently thinking of engineering a "news" feature for each company and extracting market performance details using NLP techniques to obtain better prediction results).</li>
<li>Cleaner forecasting pipeline - The current forecasting process involves the generation of a DataFrame devoted to dates and having to send it over to the model. This seems a bit inefficient. Could probably return the model instead and call the model using the date of interest</li>
<li>Adding more models</li>
<li>App styling</li>


## Acknowledgements
* [Stock Prediction Models](https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/)


##
<small><i>Note that at this stage this application is merely a proof of concept.</i></small>
