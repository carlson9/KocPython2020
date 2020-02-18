import tweepy
auth = tweepy.OAuthHandler('your consumer key', '')
auth.set_access_token('your consumer secret', '')    
api = tweepy.API(auth)

#See rate limit
api.rate_limit_status()



#Get some users
mike_ward = api.get_user('3876')

#How many favorites does he have?
mike_ward.favourites_count

#Who does Mike follow?
mikes_friends = api.friends(id=mike_ward.screen_name)
for f in mikes_friends:
  #Note I am handling UTF encoded strings so I convert them to ASCII-compatible for macs
    print("{0}".format(f.screen_name.encode('ascii', 'ignore')))
        
mikes_friends = api.friends(id=mike_ward.screen_name)
for f in mikes_friends:
  #Note I am handling UTF encoded strings for linux
        print("{0}".format(f.screen_name.encode('utf', 'ignore')))
        
        
#or get info from a screen name
gary_king = api.get_user('kinggary')
gary_friends = api.friends(id=gary_king.screen_name)
for f in gary_friends:
  #Note I am handling UTF encoded strings so I convert them to ASCII-compatible for macs
    print("{0}".format(f.screen_name.encode('ascii', 'ignore')))


import time
from datetime import timedelta

followers = api.followers_ids('davidgcarlson') # Extract IDs for my followers.
followers_count = 0 # Creating baseline of 0 followers.
i=0
while i<len(followers): # Code below loops through all users who follow me, and continues to update who stored as the "most_followed" as loop runs.
    try:
        user = api.get_user(followers[i])
        if user.followers_count > followers_count:
            followers_count = user.followers_count
            most_followed = str(user.name)
        i+=1
    except: time.sleep(.25) # Makes request every 0.25 seconds. Should we hit the limit, waits 0.25 before making another request. Permits for loop to remain active until limit is reset.


followed = api.friends_ids('mcdickenson') # Extract IDs for those users Matt is following.
i = 0
max_tweets = 0 # Creating baseline for number of tweets.
while i<len(followed): # Code below counts total tweets of each followed
    try:
        user = api.get_user(followed[i])
        tweets = user.statuses_count
        if max_tweets < tweets:
            max_tweets = tweets
            most_active = str(user.name)
        i+=1
    except: time.sleep(.25) # Makes request every 0.25 seconds. Should we hit the limit, waits 0.25 before making another request. Permits for loop to remain active until limit is reset.

print(most_followed)
print(most_active)

#TODO: pick a user that is not too active. create a network of followers, followers of the followers, etc. until you find the same user - note how many levels you have to go down

