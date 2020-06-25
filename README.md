# stockastic

<h3>Random Personal Notes/Webliography</h3>

<ul>
  <li>Type "source env/Scripts/activate" to activate the virtual environment</li>
  <li>Template files should have ".html.jinja" as their extensions. Figured that out from <a href = "https://jinja.palletsprojects.com/en/2.11.x/templates/">here</a></li>
  <li>Set threaded=False in app.run because of a tf-flask issue. Figured that out from <a href="https://stackoverflow.com/questions/58015489/flask-and-keras-model-error-thread-local-object-has-no-attribute-value">here</a></li>
  <li><a href="https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233">LSTM stuff</a></li>
  <li>My reason for picking the adam optimizer in the LSTM model can be found in this paper <a href="https://dl.acm.org/doi/pdf/10.1145/3374587.3374622">here</a></li>
</ul>

<h3>TODOs</h3>

<ul>
  <li>Figure out custom date for fetching past 5 years stock data instead of hard coding the date into the URL</li>
  <li>Implement Feature Engineering for the Linear Regression model</li>
  <li>Allow the user to select which models to display in the models route</li>
  <li>Document how to deal with dates while using a linear model for model.fit()</li>
  <li>Add hyperlinks to "read more about x model here"</li>
  <li>Display the error results (eg. RMSE etc.) in the customize model page</li>
</ul>
