import praw
from prawcore.exceptions import NotFound, Forbidden
import pandas as pd
from datetime import datetime, timedelta
import os
import json

def authenticate_reddit():
    f = open('../../reddit_credentials.json')
    credentials = json.load(f)
    f.close()
    return praw.Reddit(
        client_id = credentials["YOUR_CLIENT_ID"],
        client_secret = credentials["YOUR_CLIENT_SECRET"],
        user_agent = credentials["YOUR_USER_AGENT"]
    )

def collect_data(reddit, subreddit_name, language, limit=1000):
    try:
        subreddit = reddit.subreddit(subreddit_name)
        posts = []
        
        for post in subreddit.hot(limit=limit):
            posts.append({
                'id': post.id,
                'title': post.title,
                'body': post.selftext,
                'score': post.score,
                'num_comments': post.num_comments,
                'created_utc': datetime.fromtimestamp(post.created_utc),
                'language': language
            })
        
        return pd.DataFrame(posts)
    except NotFound:
        print(f"Subreddit r/{subreddit_name} not found. Skipping...")
        return pd.DataFrame()
    except Forbidden:
        print(f"Access to r/{subreddit_name} is forbidden. Skipping...")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while collecting data from r/{subreddit_name}: {str(e)}")
        return pd.DataFrame()

def main():
    reddit = authenticate_reddit()
    
    subreddits = {
        'climatechange': 'en',
        'ClimateActionPlan': 'en',
        'climateskeptics': 'en',
        'Klimawandel': 'de',
        'cambioclimatico': 'es',
        'changementclimatique': 'fr'
    }
    
    all_data = pd.DataFrame()
    
    for subreddit, language in subreddits.items():
        print(f"Collecting data from r/{subreddit}...")
        subreddit_data = collect_data(reddit, subreddit, language)
        if not subreddit_data.empty:
            all_data = pd.concat([all_data, subreddit_data], ignore_index=True)
    
    if all_data.empty:
        print("No data collected. Please check your subreddit list and try again.")
        return

    # Create data directory if it doesn't exist
    os.makedirs('../../data/raw', exist_ok=True)
    
    # Save the collected data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'../../data/raw/reddit_climate_data_{timestamp}.csv'
    all_data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    print(f"Total posts collected: {len(all_data)}")

if __name__ == "__main__":
    main()