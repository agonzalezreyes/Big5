import tweepy
from tweepy import OAuthHandler
import re

consumer_key = "CONSUMER KEY HERE"
consumer_secret = "CONSUMER SECRET HERE"
access_key = "ACCESS KEY HERE"
access_secret = "ACCESS SECRET HERE"

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

def clean(user):
    try:
        if '@' in user:
            usr = user.lstrip('@')
            return usr
    except TypeError:
        print "TypeError def clean(user)"
    return user

def verify(user):
    user = clean(user)
    try:
        u = api.get_user(user)
        print (u.screen_name)
        return True
    except Exception as err:
        print "verify(user) error: " + str(err)
        return False

def get_tweets(username):
    stuff = api.user_timeline(screen_name=username, count=300, include_rts=False)
    tweets = []
    for t in stuff:
        tweet_str = t.text
        tweet_str = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet_str)
        tweet_str = re.sub(r'@\w+', '*PROPNAME*', tweet_str)
        if not tweet_str: # empty strings should not be added
            continue
        tweets.append(tweet_str)
    return tweets
