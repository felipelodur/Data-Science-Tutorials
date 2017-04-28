import tweepy
from textblob import TextBlob

def createAPI( ):
    #from Twitter dev    
    consumer_key = 'YOUR_KEY'
    consumer_secret = 'YOUR_KEY'
    access_token = 'YOUR_KEY'
    access_token_secret = 'YOUR_KEY'
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    return tweepy.API(auth)

def printTweets( tweets ):
    for tweet in tweets:
        print(tweet.text)
        print( )

def printTweetsSentiment( tweets ):
    for tweet in tweets:
        print(tweet.text)
        analysis = TextBlob(tweet.text)
        print(analysis.sentiment)
        
def main():
    api = createAPI()
    public_tweets = api.search('#nintendo', count=100)
    printTweets( public_tweets )

main( )