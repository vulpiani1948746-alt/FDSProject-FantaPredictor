from bs4 import BeautifulSoup as soup
from curl_cffi import requests # CHANGED: Using curl_cffi instead of standard requests
import pandas as pd
import time
import io
import os
import sys

'''
This program will get summary player data for each game played in the top 5 
European football leagues from the website fbref.com
'''

# Standard headers are still good practice
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

def get_data_info(season, league):    
    if league == 'Serie-A':
        league_id = '11'

    url = f'https://fbref.com/en/comps/{league_id}/{season}/schedule/{season}-{league}-Scores-and-Fixtures'
    return url


def get_fixture_data(url, season, folder, league = "Serie-A"):
    print(f'Getting fixture data from: {url}')
    
    try:
        response = requests.get(url, headers=HEADERS, impersonate="chrome")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit()

    if response.status_code != 200:
        print(f"Error: Status {response.status_code} - You are blocked.")
        sys.exit()
    
    fixturedata = pd.DataFrame([])
    
    # Fix for pandas FutureWarning
    tables = pd.read_html(io.StringIO(response.text))
    
    # get fixtures
    fixtures = tables[0][['Wk', 'Day', 'Date', 'Time', 'Home', 'Away', 'xG', 'xG.1', 'Score']].dropna()
    fixtures['season'] = url.split('/')[6]
    fixturedata = pd.concat([fixturedata,fixtures])
    
    # assign id for each game
    fixturedata["game_id"] = fixturedata.index
    
    # export to csv file
    fixturedata.reset_index(drop=True).to_csv(os.path.join(folder, f'{league.lower()}_{season.lower()}_fixture_data.csv'), 
        header=True, index=False, mode='w')
    print('Fixture data collected...')


def get_match_links(url, league = "Serie-A"):   
    print('Getting match links...')
    match_links = []
    
    # --- CHANGE: Using impersonate="chrome" ---
    response = requests.get(url, headers=HEADERS, impersonate="chrome")
    
    links = soup(response.content, "html.parser").find_all('a')
    
    key_words_good = ['/en/matches/', f'{league}']
    for l in links:
        href = l.get('href', '')
        if all(x in href for x in key_words_good):
            if 'https://fbref.com' + href not in match_links:                
                match_links.append('https://fbref.com' + href)
    
    print(f"Found {len(match_links)} matches.")
    return match_links


def player_data(match_links, season, folder, league = "Serie-A"):
    player_data = pd.DataFrame([])
    
    for count, link in enumerate(match_links):
        print(f"Scraping match {count+1}/{len(match_links)}: {link}")
        try:
            # --- CHANGE: Using impersonate="chrome" ---
            response = requests.get(link, headers=HEADERS, impersonate="chrome")
            
            if response.status_code == 429:
                print("Rate limit hit (429). Sleeping for 60 seconds...")
                time.sleep(60)
                # Retry once
                response = requests.get(link, headers=HEADERS, impersonate="chrome")

            if response.status_code != 200:
                print(f"Skipping match due to error {response.status_code}")
                continue

            # Load HTML into pandas via io.StringIO
            tables = pd.read_html(io.StringIO(response.text))
            
            # Clean multi-level indices if they exist
            for table in tables:
                try:
                    if isinstance(table.columns, pd.MultiIndex):
                        table.columns = table.columns.droplevel()
                except Exception:
                    continue

            # Function to find tables safely. 
            # Note: Index numbers (tables[3]) change often on FBref. 
            # Ideally we'd search by ID, but keeping your logic for now:
            
            try:
                # FBref table indices can vary. Usually summary stats are near the top.
                # If this fails, we might need to search tables by ID attributes
                df_home = tables[3] 
                df_gk_home = tables[9]
                df_away = tables[10]
                df_gk_away = tables[16]
            except IndexError:
                # Fallback or skip if tables aren't found in expected slots
                print(" Tables not found in expected index positions. Skipping.")
                continue

            # Merge logic
            # (Keeping your original logic, but be aware merge keys must exist)
            try:
                t1 = pd.merge(df_home, df_gk_home, on=['Player', 'Nation', 'Age', 'Min'], how='outer').iloc[:-1]
                t1 = t1.assign(home=1, game_id=count)
                
                t2 = pd.merge(df_away, df_gk_away, on=['Player', 'Nation', 'Age', 'Min'], how='outer').iloc[:-1]
                t2 = t2.assign(home=0, game_id=count)
                
                player_data = pd.concat([player_data, pd.concat([t1,t2]).reset_index(drop=True)])
            except KeyError as e:
                print(f"Merge error: {e}")
                continue

            # Save progress incrementally (safer than saving only at end)
            if count % 5 == 0:
                player_data.to_csv(f'{league.lower()}_{season.lower()}_player_data.csv', index=False)

        except Exception as e:
            print(f'{link}: error {e}')
        time.sleep(1) 
    
    # Final save
    player_data.to_csv(os.path.join(folder, f'{league.lower()}_{season.lower()}_player_data.csv'), header=True, index=False)


def main_scraper(season, folder, league = 'Serie-A'): 
    url = get_data_info(season, league)
    get_fixture_data(url, season, folder, league)
    match_links = get_match_links(url, league)
    player_data(match_links, season, folder, league)

    print('Data collected!')
    