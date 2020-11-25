# Semantic Search Engine

Please create a ```Python3.7``` virtual environment, and install all the dependencies with the ```requirements.txt``` file.

After which the search engine can be started with ```python app.py```

At the time of deployment, change the url in ```templates/search.html``` and ```templates/result.html``` at line 50 to the ```SERVER_URL/PORT/auto_complete```

***Note:***
The flask app is hosted with the help of waitress API(For Windows based servers), refer to [this link](https://docs.pylonsproject.org/projects/waitress/en/stable/arguments.html#arguments) for documentation.