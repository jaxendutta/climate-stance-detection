import praw
from prawcore.exceptions import NotFound, Forbidden
import pandas as pd
from datetime import datetime
import os
import configparser
import json
from time import sleep
from langdetect import detect
from tqdm import tqdm
from halo import Halo

# Global Configuration
SUBREDDITS = {
    'en': [
        'climate',
        'climatechange',
        'ClimateActionPlan',
        'climateskeptics',
        'ClimateOffensive',
        'ClimateCrisis',
        'environment',
        'environmental_science'
    ],
    'de': [
        'Klimawandel',
        'umwelt_de'
    ],
    'es': [
        'CambioClimatico',
        'Medioambiente'
    ],
    'fr': [
        'ecologie'
    ],
    'it': [
        'cambiamentoclimatico'
    ]
}

def authenticate_reddit():
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    return praw.Reddit(
        client_id=config['Reddit']['client_id'],
        client_secret=config['Reddit']['client_secret'],
        user_agent=config['Reddit']['user_agent']
    )

def verify_language(text, expected_language):
    """
    Verify if text matches expected language.
    Returns a tuple of (is_verified, verification_status)
    """
    if not text or len(text.strip()) < 50:
        return (True, 'too_short')
    try:
        detected = detect(text)
        return (detected == expected_language, 'verified')
    except:
        return (True, 'detection_failed')

def collect_data(reddit, subreddit_name, language, limit=None, progress_index=None):
    try:
        subreddit = reddit.subreddit(subreddit_name)
        posts = []
        
        print(f"\n[{progress_index}] r/{subreddit_name}")
        
        # Collect from multiple listing types
        listing_methods = {
            'hot': subreddit.hot(limit=None),
            'new': subreddit.new(limit=None),
            'top': subreddit.top(limit=None, time_filter='all')
        }
        
        for method_name, listing in listing_methods.items():
            method_posts = []
            
            spinner = Halo(
                text=f'({method_name}): 0 posts',
                spinner='dots',
                color='cyan'
            )
            spinner.start()
            
            try:
                for post in listing:
                    if any(p['id'] == post.id for p in posts):
                        continue
                        
                    post_data = {
                        'id': post.id,
                        'title': post.title,
                        'body': post.selftext,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'language': language,
                        'subreddit': subreddit_name,
                        'listing_type': method_name,
                        'collected_at': datetime.now()
                    }
                    
                    post_data['title_length'] = len(post.title)
                    post_data['body_length'] = len(post.selftext)
                    
                    combined_text = f"{post.title} {post.selftext}"
                    is_verified, verification_status = verify_language(combined_text, language)
                    post_data['language_verified'] = is_verified
                    post_data['verification_status'] = verification_status
                    post_data['total_length'] = len(combined_text.strip())
                    
                    method_posts.append(post_data)
                    posts.append(post_data)
                    
                    spinner.text = f'({method_name}): {len(method_posts)} posts'
                    
                    if len(method_posts) % 100 == 0:
                        sleep(1)
                
                spinner.succeed(f'({method_name}): {len(method_posts)} posts collected')
                
            except Exception as e:
                spinner.stop()
                continue
            finally:
                spinner.stop()
            
            sleep(2)
        
        if posts:
            print(f"SUCCESS: {len(posts)} posts collected from r/{subreddit_name}!")
            return pd.DataFrame(posts)
        else:
            print(f"✖ [Error] for r/{subreddit_name}: No posts could be collected")
            return pd.DataFrame()
    
    except NotFound:
        print(f"✖ [Error] for r/{subreddit_name}: Subreddit not found")
    except Forbidden:
        print(f"✖ [Error] for r/{subreddit_name}: Access forbidden")
    except Exception as e:
        print(f"✖ [Error] for r/{subreddit_name}: {str(e)}")
    return pd.DataFrame()

def generate_collection_stats(data, start_time, end_time):
    """Generate statistics about the collected data"""
    duration = end_time - start_time
    
    verification_stats = data['verification_status'].value_counts().to_dict()
    
    length_stats = {
        'very_short_posts': len(data[data['total_length'] < 50]),
        'short_posts': len(data[(data['total_length'] >= 50) & (data['total_length'] < 200)]),
        'medium_posts': len(data[(data['total_length'] >= 200) & (data['total_length'] < 1000)]),
        'long_posts': len(data[data['total_length'] >= 1000])
    }
    
    return {
        'collection_timing': {
            'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S"),
            'duration_seconds': duration.total_seconds(),
            'duration_minutes': duration.total_seconds() / 60,
            'duration_hours': duration.total_seconds() / 3600
        },
        'total_posts': len(data),
        'by_language': data['language'].value_counts().to_dict(),
        'by_subreddit': data['subreddit'].value_counts().to_dict(),
        'verification_stats': verification_stats,
        'length_distribution': length_stats,
        'avg_title_length': data['title_length'].mean(),
        'avg_body_length': data['body_length'].mean(),
        'avg_score': data['score'].mean(),
        'avg_comments': data['num_comments'].mean(),
        'language_verification_rate': data['language_verified'].mean(),
    }

def main():
    print("\n=== Reddit Climate Discussion Data Extraction ===")
    print("Starting data collection process. Please be patient...")
    
    start_time = datetime.now()
    reddit = authenticate_reddit()
    
    all_data = pd.DataFrame()
    total_subreddits = sum(len(subs) for subs in SUBREDDITS.values())
    processed = 0
    
    for language, subreddit_list in SUBREDDITS.items():
        for subreddit in subreddit_list:
            processed += 1
            progress_str = f"{processed}/{total_subreddits}"
            subreddit_data = collect_data(reddit, subreddit, language, 
                                        progress_index=progress_str)
            if not subreddit_data.empty:
                all_data = pd.concat([all_data, subreddit_data], ignore_index=True)
    
    if all_data.empty:
        print("ERROR: No data collected. Please check your subreddit list and try again.")
        return

    end_time = datetime.now()
    
    os.makedirs('data/raw', exist_ok=True)
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    filename = f'data/raw/reddit_climate_data_{timestamp}.csv'
    all_data.to_csv(filename, index=False)
    
    stats = generate_collection_stats(all_data, start_time, end_time)
    stats_filename = f'data/raw/collection_stats_{timestamp}.json'
    with open(stats_filename, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print("\n" + "=" * 60)
    print("Collection completed successfully!")
    print(f"\nData saved to: {filename}")
    print(f"Statistics saved to: {stats_filename}")
    print("\nCollection Summary:")
    print(f"Total posts collected: {stats['total_posts']}")
    print("\nPosts by language:")
    for lang, count in stats['by_language'].items():
        print(f"{lang}: {count}")
    print("\nVerification status distribution:")
    for status, count in stats['verification_stats'].items():
        print(f"{status}: {count}")
    print("\nLength distribution:")
    for category, count in stats['length_distribution'].items():
        print(f"{category}: {count}")
    print(f"\nLanguage verification rate: {stats['language_verification_rate']*100:.2f}%")
    print(f"Collection duration: {stats['collection_timing']['duration_minutes']:.2f} minutes")

if __name__ == "__main__":
    main()