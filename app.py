from flask import Flask, request
from dotenv import load_dotenv
from pathlib import Path
import os
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
    app.logger.info('HERE')
    endDate = datetime.datetime.now()
    if mode == 'hour':
        timedelta = datetime.timedelta(hours=1)
    elif mode =='week':
        timedelta = datetime.timedelta(weeks=1)
    startDate = endDate - timedelta
    tweets = get_tweets_for_user_within_dates(api, username, 
                                              startDate, endDate,
                                              count, too_many_tweets,
                                              max_attempts)
    startDate_endDate_str = startDate.strftime("%Y-%m-%d_%H:%M") \
                            + '_' + endDate.strftime("%Y-%m-%d_%H:%M")
    if not silent:
        n = len(tweets)
        print('from: {}, to: {}, number of tweets: {}'.format(startDate,
                                                              endDate, 
                                                              n))
        if n > 0:
            print(utc_to_local(tweets[0].created_at), utc_to_local(tweets[-1].created_at))
    id_lst, created_at_lst, full_text_lst, response_lst = [], [], [], []
    for tweet in tqdm(tweets, disable=silent):
        id_lst.append(tweet.id)
        created_at_lst.append(utc_to_local(tweet.created_at))
        full_text_lst.append(tweet.full_text.replace('\n', ' '))
        response_lst.append(get_response(my_analyzer, tweet.full_text))
    data = pd.DataFrame({
        'id': id_lst,
        'created_at': created_at_lst, 
        'full_text': full_text_lst,
        'response': response_lst,
    })
    data.to_csv('./results/user.csv', index=False)
    print(data, flush=True)
    return data





def send_message(body, from_="+14193860121", to="+17814285958"):
    message = client.messages \
                .create(
                     body=body,
                     from_ = from_,
                     to=to
                 )
    

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


@app.route("/")
def hello_world():
    
    print('Hi', flush=True)
    mode = 'hour'
    username = 'elonmusk'
    count = 50
    too_many_tweets = 1000
    max_attempts = 10
    silent = False
    d = program(api, username, silent, mode, count, too_many_tweets, max_attempts)

    # scheduler = BlockingScheduler()
    # _ = scheduler.add_job(lambda: program(api, username, silent, mode,
    #                                    count, too_many_tweets, 
    #                                   max_attempts), 
    #                  'interval', minutes=2)
    # scheduler.start()

    boo = True

    if boo:
        send_message('Test')
        
    return 'done'

# @app.route("/get_tweet")
# def hello_world():
#     return "<p>Hello, World!</p>" 

if __name__ == '__main__':
    app.run()



# Get tweets every hour
# Run sentiment analysis
# If SOME_METRIC < THRESHOLD:
# TRIGGER MESSAGE
