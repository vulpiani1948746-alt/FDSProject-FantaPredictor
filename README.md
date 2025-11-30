### FDS Project
The project consists in the creation of a deep learning model that, given the statistics about the last N matches, predicts the  grade of a player in Fantasy Football for the next match.

## What did we do so far?
* We scraped the data from fbref, applying the code present at https://github.com/adamcorren/fbref_football_player_data_scraper. We encountered a problem because the website was blocking us after recognizing we were using a bot. To solve the problem we changed the code to impersonate a real user that accesses the website (used gemini pro to do it). In this way, we were able to extract all the statistics about every player and every team for every match of the last 8 seasons.
* We scraped the Fantasy Football data from Fantacalcio.it by downloading the excel files the website provides for each game of the season. The problem here was that the website requires access to it to download the data so we had to copy the cookie and insert its information to automatically scrape the website. In this way we obtained data about the grades of each player for each match.
* We merged the data into a unique pandas dataframe by addressing the problems of some footbalers' names that are spelled differently in the two datasets (we manually corrected some of them), we did some feature engineering (we added some features extracted from the ones available) and we deleted the highly correlated features.
* Once the features were complete we trained an LSTM that "looks" at the last 6 matches and predicts the fantasy football grade of the player for the next match. It is made of the normal LSTM and it adds a dense layer that takes in input the "future" features that are the ones known about the match (the strength of the opposing team, is it at home or not).
* Later we did feature selection (with permutation importance) and hyperparameter tuning (for the value of N and the normal hyperparameters of the NN)
* We created an xgboost model that predicts the result of a match and integrated this prediction in the dense layer of the LSTM to predict the grade of the footballer.

## The next steps:
* Add the embedding layer (so the NN always knows who is the player) and do hyperparameter tuning
* Create the 1D CNN model from the time series and evaluate it


## References
https://github.com/adamcorren/fbref_football_player_data_scraper (for the data scraping of fbref)
gupta, a. (2019). machine learning for fantasy football prediction. stanford cs229 project report. (for taking ispiration, especially for which are the most important feature to use)
Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football betting market. Journal of the Royal Statistical Society: Series C (Applied Statistics), 46(2), 265-280. (statistical method for predicting match results)
Deep Learning and Transfer Learning Architectures for English Premier League Player Performance Forecasting (for the CNN architecture)
