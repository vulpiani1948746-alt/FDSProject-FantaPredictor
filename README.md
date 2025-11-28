### FDS Project
The project consists in the creation of a deep learning model that, given the statistics about the last N matches, predicts the average grade of a player in Fantasy Football for the next M matches.

## What did we do so far?
* We scraped the data from fbref, applying the code present at https://github.com/adamcorren/fbref_football_player_data_scraper. We encountered a problem because the website was blocking us after recognizing we were using a bot. To solve the problem we changed the code to impersonate a real user that accesses the website (used gemini pro to do it). In this way, we were able to extract all the statistics about every player and every team for every match of the last 8 seasons.
* We scraped the Fantasy Football data from Fantacalcio.it by downloading the excel files the website provides for each game of the season. The problem here was that the website requires access to it to download the data so we had to copy the cookie and insert its information to automatically scrape the website. In this way we obtained data about the grades of each player for each match.
* We merged the data into a unique pandas dataframe by addressing the problems of some footbalers' names that are spelled differently in the two datasets.

## The next steps:
* Create the LSTM model from the time series and evaluate it
* Create the GRU model from the time series and evaluate it
* Create the 1D CNN model from the time series and evaluate it
* Create a Random Forest model to have as a second baseline?


## References
https://github.com/adamcorren/fbref_football_player_data_scraper
