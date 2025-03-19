import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from googleapiclient.discovery import build

# Ladda ner NLTK-resurser
nltk.download('vader_lexicon')

# API-inställningar (ange din egen YouTube API-nyckel här)
API_KEY = "DIN_YOUTUBE_API_NYCKEL"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Ange video-ID för Kevin Hart och Donald Trump intervjuer
VIDEO_IDS = {
    "Kevin Hart": "VIDEO_ID_KEVIN_HART",
    "Donald Trump": "VIDEO_ID_DONALD_TRUMP"
}

def get_youtube_comments(video_id, max_comments=2000):
    """Hämtar kommentarer från en YouTube-video via API."""
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments[:max_comments]

def classify_comments(comments, keywords):
    """Klassificerar kommentarer som relevanta eller irrelevanta baserat på nyckelord."""
    relevant_comments = [c for c in comments if any(word in c.lower() for word in keywords)]
    return relevant_comments

def analyze_sentiment(comments):
    """Utför sentimentanalys med VADER."""
    sia = SentimentIntensityAnalyzer()
    sentiment_results = {'Positive': 0, 'Neutral': 0, 'Negative': 0}

    for comment in comments:
        score = sia.polarity_scores(comment)['compound']
        if score >= 0.05:
            sentiment_results['Positive'] += 1
        elif score <= -0.05:
            sentiment_results['Negative'] += 1
        else:
            sentiment_results['Neutral'] += 1

    return sentiment_results

def visualize_results(results, title):
    """Visualiserar sentimentfördelning i ett stapeldiagram."""
    categories = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(8, 5))
    plt.bar(categories, values, color=['green', 'gray', 'red'])
    plt.xlabel("Sentiment")
    plt.ylabel("Antal kommentarer")
    plt.title(f"Sentimentanalys för {title}")
    plt.show()

# Nyckelord för relevanta kommentarer
KEYWORDS = {
    "Kevin Hart": ["comedian", "stand-up", "humor", "funny", "joke"],
    "Donald Trump": ["president", "politics", "debate", "election", "policy"]
}

# Hämta och analysera kommentarer för båda videorna
for name, video_id in VIDEO_IDS.items():
    print(f"Hämtar kommentarer för {name}...")
    comments = get_youtube_comments(video_id)
    relevant_comments = classify_comments(comments, KEYWORDS[name])
    sentiment_results = analyze_sentiment(relevant_comments)
    visualize_results(sentiment_results, name)
