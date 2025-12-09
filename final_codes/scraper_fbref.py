from bs4 import BeautifulSoup as soup
from curl_cffi import requests 
import pandas as pd
import time
import io
import os
import sys

'''
This program will get summary player data for each game played in the top 5 
European football leagues from the website fbref.com
'''

# Standard headers to mimic a legitimate browser request
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

def get_data_info(season, league):    
    """
    Constructs the URL for the specific league and season schedule page on FBref.
    """
    if league == 'Serie-A':
        league_id = '11' # FBref internal ID for Serie A

    # Construct the URL pattern: /comps/{id}/{season}/schedule/{season}-{league}-Scores-and-Fixtures
    url = f'https://fbref.com/en/comps/{league_id}/{season}/schedule/{season}-{league}-Scores-and-Fixtures'
    return url


def get_fixture_data(url, season, folder, league="Serie-A"):
    """
    Scrapes the main schedule/fixture table from the league page.
    This table contains general match info: Date, Teams, Scores, xG (Expected Goals).
    """
    print(f'Getting fixture data from: {url}')
    
    try:
        # Use impersonate="chrome" to avoid 403 Forbidden errors common with standard requests
        response = requests.get(url, headers=HEADERS, impersonate="chrome")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit()

    if response.status_code != 200:
        print(f"Error: Status {response.status_code} - You are blocked.")
        sys.exit()
    
    fixturedata = pd.DataFrame([])
    
    # Read HTML tables directly into pandas
    tables = pd.read_html(io.StringIO(response.text))
    
    # Extract specific columns from the first table found (the fixtures table)
    fixtures = tables[0][['Wk', 'Day', 'Date', 'Time', 'Home', 'Away', 'xG', 'xG.1', 'Score']].dropna()
    fixtures['season'] = url.split('/')[6] # Add season identifier
    fixturedata = pd.concat([fixturedata,fixtures])
    
    # Create a unique game_id based on the dataframe index
    fixturedata["game_id"] = fixturedata.index
    
    # Save the general fixture data to CSV
    fixturedata.reset_index(drop=True).to_csv(os.path.join(folder, f'{league.lower()}_{season.lower()}_fixture_data.csv'), 
        header=True, index=False, mode='w')
    print('Fixture data collected...')


def get_match_links(url, league="Serie-A"):   
    """
    Parses the schedule page HTML to find links to every individual match report.
    Returns a list of URLs for specific matches.
    """
    print('Getting match links...')
    match_links = []
    
    response = requests.get(url, headers=HEADERS, impersonate="chrome")
    
    # Use BeautifulSoup to parse HTML links <a>
    links = soup(response.content, "html.parser").find_all('a')
    
    # Filter links that look like match reports ('/en/matches/') and contain the league name
    key_words_good = ['/en/matches/', f'{league}']
    for l in links:
        href = l.get('href', '')
        if all(x in href for x in key_words_good):
            # Ensure unique links and append base domain
            if 'https://fbref.com' + href not in match_links:                
                match_links.append('https://fbref.com' + href)
    
    print(f"Found {len(match_links)} matches.")
    return match_links


def player_data(match_links, season, folder, league="Serie-A"):
    """
    Iterates through every match link, scrapes detailed player stats, and consolidates them.
    Warning: This function sleeps frequently to avoid rate limits.
    """
    player_data = pd.DataFrame([])
    
    for count, link in enumerate(match_links):
        print(f"Scraping match {count+1}/{len(match_links)}: {link}")
        try:
            response = requests.get(link, headers=HEADERS, impersonate="chrome")
            
            # Handle Rate Limiting (HTTP 429) manually
            if response.status_code == 429:
                print("Rate limit hit (429). Sleeping for 60 seconds...")
                time.sleep(60)
                response = requests.get(link, headers=HEADERS, impersonate="chrome")

            if response.status_code != 200:
                print(f"Skipping match due to error {response.status_code}")
                continue

            # Parse all tables on the match page
            tables = pd.read_html(io.StringIO(response.text))
            
            # Fix multi-level column headers (common in FBRef tables)
            for table in tables:
                try:
                    if isinstance(table.columns, pd.MultiIndex):
                        table.columns = table.columns.droplevel()
                except Exception:
                    continue
            
            try:
                # Specific indices for standard FBRef match pages (may break if site layout changes)
                df_home = tables[3] 
                df_gk_home = tables[9]
                df_away = tables[10]
                df_gk_away = tables[16]
            except IndexError:
                print(" Tables not found in expected index positions. Skipping.")
                continue

            try:
                # Merge outfield player stats with Goalkeeper stats for Home Team
                t1 = pd.merge(df_home, df_gk_home, on=['Player', 'Nation', 'Age', 'Min'], how='outer').iloc[:-1]
                t1 = t1.assign(home=1, game_id=count)
                
                # Merge for Away Team
                t2 = pd.merge(df_away, df_gk_away, on=['Player', 'Nation', 'Age', 'Min'], how='outer').iloc[:-1]
                t2 = t2.assign(home=0, game_id=count)
                
                # Combine both teams into the main dataframe
                player_data = pd.concat([player_data, pd.concat([t1,t2]).reset_index(drop=True)])
            except KeyError as e:
                print(f"Merge error: {e}")
                continue

            # Periodic checkpoint save (every 5 matches)
            if count % 5 == 0:
                player_data.to_csv(f'{league.lower()}_{season.lower()}_player_data.csv', index=False)

        except Exception as e:
            print(f'{link}: error {e}')
        
        # Respectful scraping delay
        time.sleep(1) 
    
    # Final Save
    player_data.to_csv(os.path.join(folder, f'{league.lower()}_{season.lower()}_player_data.csv'), header=True, index=False)


def main_scraper(season, folder, league='Serie-A'): 
    """
    Orchestrator function.
    """
    url = get_data_info(season, league)
    get_fixture_data(url, season, folder, league)
    match_links = get_match_links(url, league)
    player_data(match_links, season, folder, league)

    print('Data collected!')