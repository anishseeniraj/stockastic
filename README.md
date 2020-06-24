# stockastic

<h3>Random Personal Notes</h3>

<ul>
  <li>Type "source env/Scripts/activate" to activate the virtual environment</li>
  <li>Template files should have ".html.jinja" as their extensions. Figured that out from <a href = "https://jinja.palletsprojects.com/en/2.11.x/templates/">here</a></li>
  <li>Set threaded=False in app.run because of a tf-flask issue. Figured that out from <a href="https://stackoverflow.com/questions/58015489/flask-and-keras-model-error-thread-local-object-has-no-attribute-value">here</a></li>
</ul>

<h3>TODOs</h3>

<ul>
  <li>Figure out custom date for fetching past 5 years stock data instead of hard coding the date into the URL</li>
  <li>Implement Feature Engineering for the Linear Regression model</li>
  <li>Document how to deal with dates while using a linear model for model.fit()</li>
  <li>Add hyperlinks to "read more about x model here"</li>
  <li>Display the error results (eg. RMSE etc.) in the customize model page</li>
</ul>
