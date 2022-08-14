# 1 Feature Linear Regression Model - Comparing Simpson views to IMDB Rating

This project is a naive implementation of gradient descent for 1 feature Linear Regression Models using numpy. This example uses [This dataset here](https://www.kaggle.com/datasets/jonbown/simpsons-episodes-2016?resource=download) to create a prediction for IMDB Ratings (x) to views (y) for the Simpsons episodes.

# Shortcomings
The data does not seem to have a high correlation between IMDB Ratings and View counts from the dataset. Therefore, predictions from this linear model do not seem to produce very usable results.

# Dependencies
Python 3.X
`pip install numpy matplotlib pandas`

To run simply execute `python index.py`