from flask import Flask, request
from dotenv import load_dotenv
from pathlib import Path
import os
import random
import requests
import twilio
from twilio.rest import Client

import pandas as pd
import tweepy
import datetime
import re
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, EntitiesOptions, EmotionOptions, ConceptsOptions, CategoriesOptions, KeywordsOptions
from tqdm import tqdm
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report
import operator
import langid
import matplotlib.pyplot as plt
from apscheduler.schedulers.blocking import BlockingScheduler

import pickle as pkl

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)
app = Flask(__name__)

# LOAD IN CONSTANTS
CONSUMER_KEY=os.environ.get('CONSUMER_KEY')
CONSUMER_SECRET=os.environ.get('CONSUMER_SECRET')
ACCESS_TOKEN=os.environ.get('ACCESS_TOKEN')
ACCESS_TOKEN_SECRET=os.environ.get('ACCESS_TOKEN_SECRET')

TWILIO_ACCOUNT_SID=os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN=os.environ.get('TWILIO_AUTH_TOKEN')

WATSON_APIKEY = os.environ.get('WATSON_APIKEY')
WATSON_URL = os.environ.get('WATSON_URL')

# Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


# Helper method that converts UTC time to local time
def utc_to_local(time):
    return time-datetime.timedelta(hours=4)

# Helper method that gets Tweets of a user within dates
def get_tweets_for_user_within_dates(api, username, startDate, endDate, 
                                     count=50, too_many_tweets=1000, 
                                     max_attempts=5):
    attempts = 0
    #initialize a list to hold all the tweepy Tweets
    alltweets = []      
    while attempts < max_attempts:
        attempts += 1
        if len(alltweets) == 0:
            #make initial request for most recent tweets
            new_tweets = api.user_timeline(screen_name=username, 
                                            count=count,
                                            tweet_mode='extended')
        else:
            #all subsequent requests use max_id to prevent duplicates
            new_tweets = api.user_timeline(screen_name=username,
                                            count=count,
                                            max_id=oldest_id,
                                            tweet_mode='extended')
        if len(new_tweets) > 0:
            attempts = 0
            #save most recent tweets
            alltweets.extend(new_tweets)
            if len(alltweets) > too_many_tweets:
                raise ValueError('Too many tweets. Select diff date range.')
            #save the id of the oldest tweet less one
            oldest_id = alltweets[-1].id-1
            oldest_date = alltweets[-1].created_at
            #check if enough tweets obtained
            if utc_to_local(oldest_date) <= startDate:
                break
    tweets = []
    for tweet in alltweets:
        if startDate <= utc_to_local(tweet.created_at) <= endDate:
            tweets.append(tweet)
    return tweets


# Helper method that conducts sentiment on text using Watson NLU service
def get_response(analyzer, text):
    response = analyzer.analyze(
        text=text,
        language='en',
        features=Features(
            categories=CategoriesOptions(
                limit=5,
                explanation=True,
            ),
            concepts=ConceptsOptions(
                limit=5,
            ),
            emotion=EmotionOptions(),
            entities=EntitiesOptions(
                limit=5,
                sentiment=True,
                emotion=True,
            ),
            sentiment=SentimentOptions(),
        ),
    ).get_result()
    return response


# Helper method that conducts sentiment on text using Watson NLU service
def get_response(analyzer, text):
    response = analyzer.analyze(
        text=text,
        language='en',
        features=Features(
            categories=CategoriesOptions(
                limit=5,
                explanation=True,
            ),
            concepts=ConceptsOptions(
                limit=5,
            ),
            emotion=EmotionOptions(),
            entities=EntitiesOptions(
                limit=5,
                sentiment=True,
                emotion=True,
            ),
            sentiment=SentimentOptions(),
        ),
    ).get_result()
    return response

# Helper method to run program
# params:

# mode: str
# pass 'week' if you want to check past week's worth of tweets.
# pass 'hour' if you want to check past hour's worth of tweets.

def program(api, username, silent=False, mode='hour',
            count=50, too_many_tweets=1000, max_attempts=5):
    
    data_path = './results/{}.csv'.format(username)
    if not os.path.isfile(data_path):
        data = pd.DataFrame(columns=['id', 'created_at', 'full_text', 'positivity',
                                     'sadness', 'joy', 'fear', 'disgust', 'anger', 
                                     'categories', 'concepts', 'entities'])    
        data.to_csv(data_path, sep=',', index=False) 
    data = pd.read_csv(data_path, sep=',')
    data['created_at'] = pd.to_datetime(data['created_at'], format='%Y-%m-%d %H:%M:%S')
    if mode == 'hour':
        endDate = datetime.datetime.now()
        timedelta = datetime.timedelta(hours=1)
        startDate = endDate - timedelta
        tweets = get_tweets_for_user_within_dates(api, username, 
                                                  startDate, endDate,
                                                  count, too_many_tweets,
                                                  max_attempts)
        results = {'sentiment': [], 'emotion': []}
        n = len(tweets)
        if n == 0:
            print('no tweets found')
            return results
        if not silent:
            print('from: {}, to: {}, number of tweets: {}'.format(startDate,
                                                                  endDate, 
                                                                  n))
        data_ids = data['id']
        positivity_lst = []
        sadness_lst, joy_lst, fear_lst, disgust_lst, anger_lst = [], [], [], [], []
        found_new_tweet = False
        for tweet in tqdm(tweets, disable=silent):
            if tweet.id in data_ids:
                continue
            found_new_tweet = True
            response = get_response(my_analyzer, tweet.full_text)
            sentiment = response["sentiment"]["document"]
            row = {"id": tweet.id, "created_at": utc_to_local(tweet.created_at),
                   "full_text": tweet.full_text.replace('\n', ' '),
                   "positivity": 0.5 if sentiment['label'] == 'neutral' else (sentiment['score'] + 1) / 2,
                   "sadness": 0, "joy": 0, "fear": 0, "disgust": 0, "anger": 0,
                   "categories": response["categories"], 
                   "concepts": response["concepts"],
                   "entities": response["entities"]}
            for emotion, confidence in response['emotion']['document']['emotion'].items():
                row[emotion] = confidence
            data = data.append(row, ignore_index=True)
            positivity_lst.append(row['positivity'])
            sadness_lst.append(row['sadness']); joy_lst.append(row['joy']); fear_lst.append(row['fear'])
            disgust_lst.append(row['disgust']); anger_lst.append(row['anger'])
        if not found_new_tweet:
            print('no new tweets found')
            return results
        data.to_csv(data_path, sep=',', index=False)
        positivity_avg = np.mean(positivity_lst)
        sadness_avg = np.mean(sadness_lst); joy_avg = np.mean(joy_lst); fear_avg = np.mean(fear_lst)
        disgust_avg = np.mean(disgust_lst); anger_avg = np.mean(anger_lst)
        for sentiment, (interval_low, interval_high) in sentiment_label_2_bin.items():
            if interval_low <= positivity_avg < interval_high:
                results['sentiment'].append(sentiment)
                break
            elif interval_high == 1 and positivity_avg == 1:
                results['sentiment'].append(sentiment)
                break
        for emotion, (interval_low, interval_high) in emotion_label_2_bin.items():
            emotion_avg = locals()[emotion+'_avg']
            if interval_low <= emotion_avg < interval_high:
                results['emotion'].append(emotion)
            elif interval_high == 1 and emotion_avg == 1:
                results['emotion'].append(emotion)
        print('tweets analyzed')
        return results
    elif mode == 'week':
        data_avg_path = './results/{}_avg.csv'.format(username)
        if not os.path.isfile(data_avg_path):
            data_avg = pd.DataFrame(columns=['start_date', 'end_date', 'positivity_avg',
                                             'sadness_avg', 'joy_avg', 'fear_avg', 
                                             'disgust_avg', 'anger_avg'])    
            data_avg.to_csv(data_avg_path, sep=',', index=False) 
        data_avg = pd.read_csv(data_avg_path, sep=',')
        endDate = round_seconds(datetime.datetime.now())
        timedelta = datetime.timedelta(weeks=1)
        startDate = endDate - timedelta
        data_week = data[data['created_at'].between(startDate, endDate)]
        data_week_avg = data_week.mean(axis=0)
        row = {'start_date': startDate, 'end_date': endDate,
            'positivity_avg': data_week_avg['positivity'],
            'sadness_avg': data_week_avg['sadness'],
            'joy_avg': data_week_avg['joy'],
            'fear_avg': data_week_avg['fear'],
            'disgust_avg': data_week_avg['disgust'],
            'anger_avg': data_week_avg['anger']}
        data_avg = data_avg.append(row, ignore_index=True)
        data_avg = data_avg.round(2)
        data_avg.to_csv(data_avg_path, sep=',', index=False)


def send_message(emotion=None, from_="+14193860121", to="+17814285958"):

    
    rndm = random.randint(0, 101)
    quotestr = quotes[rndm]['quote'] + ' - ' + quotes[rndm]['author']
    rndm2 = random.randint(0, 239)
    URLstr = tracks['URL'][rndm2] + ' \n'

    msg = ''
    if not emotion:
        msg = 'Hey there, you seem to be feeling a little down, just wanted to check in and drop a meditation link if that helps <3 ' + URLstr + quotestr
    else:
        emotionstr = ' and '.join(emotion)
        msg = 'Hey there, you seem to be feeling a little ' + emotionstr + ', just wanted to check in and drop a meditation link if that helps <3 ' + URLstr + quotestr
        

    message = client.messages \
                .create(
                     body=msg,
                     from_ = from_,
                     to=to
                 )
    print(msg, flush=True)


# INIT
auth = tweepy.OAuthHandler(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
authenticator = IAMAuthenticator(WATSON_APIKEY)
my_analyzer = NaturalLanguageUnderstandingV1(
    version='2020-08-01',
    authenticator=authenticator
)
my_analyzer.set_service_url(WATSON_URL)

# QUOTES
quotes = pd.read_json('quotes.json').quotes
tracks = pd.read_csv('links.csv')

# Apply thresholds for sentiment
sentiment_label_2_bin = {'negative': (0, 0.43), 'neutral': (0.43, 0.7), 'positive': (0.7, 1)}
# Apply thresholds for emotion
emotion_label_2_bin = {'joy': (0.58, 1), 
                    'sadness': (0.55, 1), 
                    'fear': (0.55, 1), 
                    'disgust': (0.55, 1), 
                    'anger': (0.55, 1)}



@app.route("/")
def hello_world():
    
    print('Hola', flush=True)
    mode = 'hour'
    username = 'MiDJs4U'
    count = 50
    too_many_tweets = 1000
    max_attempts = 10
    silent = False
    d = program(api, username, silent, mode, count, too_many_tweets, max_attempts)
    print(d, flush=True)

    if not d or (d['sentiment'][0] == 'neutral' or d['sentiment'][0] == 'positive'):
        return 'done with no message'

    send_message(d['emotion'])
    
    

    # scheduler = BlockingScheduler()
    # _ = scheduler.add_job(lambda: program(api, username, silent, mode,
    #                                    count, too_many_tweets, 
    #                                   max_attempts), 
    #                  'interval', minutes=2)
    # scheduler.start()
        
    return 'done with message'

# @app.route("/get_tweet")
# def hello_world():
#     return "<p>Hello, World!</p>" 

if __name__ == '__main__':
    app.run(debug=True)