import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

import config
import json
import re

#Download Tweets
class MyListener(StreamListener):

    def __init__(self, data_dir, filename):
        self.outfile = "%s/%s.json" % (data_dir, filename)

    def on_data(self, data):
        try:
            with open(self.outfile, 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        return True

auth = OAuthHandler(config.consumer_key, config.consumer_secret)
auth.set_access_token(config.access_token, config.access_secret)
api = tweepy.API(auth)

twitter_stream = Stream(auth, MyListener('data', 'twitter'))
twitter_stream.filter(languages=["en"],locations=[-124.848974,24.396308,-66.885444,49.384358])

#Parse Tweets
tweets_data_path = 'data/twitter.json'
tweets_data = []
tweets_file = open(tweets_data_path,'r')

for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue

def preprocess(s):
    eng = "[a-zA-Z]+"
    tokens = re.findall(eng,s)
    tokens = [token.lower() for token in tokens]
    return tokens

def get_coordinates(x):
    if x:
        return x['coordinates']
    return x

tweets_df = pd.DataFrame()
tweets_df['id'] = map(lambda tweet : str(tweet.get('id')), tweets_data)
tweets_df['text'] = map(lambda tweet : tweet.get('text','').encode("utf-8"), tweets_data)
tweets_df['coordinates'] = map(lambda tweet : tweet.get('coordinates',None), tweets_data)
tweets_df['text'] = tweets_df['text'].map(preprocess)
tweets_df['coordinates'] = tweets_df['coordinates'].map(get_coordinates)
data = tweets_df[tweets_df['id'] != 'None']

#Get Tweets
tweets = list(data['text'])

#TF-IDF
print ('TFIDF Feature...')
max_features = 1000
vectorizer = TfidfVectorizer(min_df=1, max_features=max_features)
tweets_tfidf = vectorizer.fit_transform(tweets)

#NMF
print ('NMF...')
n_components = 2
nmf = NMF(n_components=n_components, random_state=1)
W = nmf.fit_transform(tweets_tfidf)
cols = ['tfidf_'+str(i) for i in range(n_components)]
tweets_nmf_df = pd.DataFrame(W,columns=cols)
tweets_df = pd.concat((data['id'],tweets_nmf_df), axis=1)

#KMeans
print ('KMeans clustering...')
n_clusters = [i for i in range(2,14)]
scores = []
for ncluster in n_clusters:
    print ('%d clusters')%ncluster
    clf = KMeans(n_clusters=ncluster, init='k-means++', n_init=40, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=0)
    model = clf.fit(tweets_df[cols])
    scores.append(model.inertia_)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(n_clusters,scores,'ro')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.savefig('clusters_nmf.png')

#Choose Best Number of Clusters
print ('Making Predication...')
n_best = 4
clf = KMeans(n_clusters=n_best, init='k-means++', n_init=40, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=0)
y_predict = clf.fit_predict(tweets_df[cols])
cluster_center = clf.cluster_centers_
tweets_df['cluster'] = y_predict
tweets_df['coordinates'] = data['coordinates']

colors = ["gray","coral","purple","red","orange","green","blue"]
fig = plt.figure()
ax = fig.add_subplot(111)
for k, col in zip(range(n_best),colors):
    members = y_predict == k
    center = cluster_center[k]
    x = list(tweets_df.loc[members,'tfidf_0'])
    y = list(tweets_df.loc[members,'tfidf_1'])
    ax.plot(x,y,'w',markerfacecolor=col,marker='.')
    ax.plot(center[0],center[1],'o',markerfacecolor=col,markeredgecolor='k',markersize=6)
plt.xlabel('reduced feature 0')
plt.ylabel('reduced feature 1')
plt.savefig('clusters_nmf_label.png')

tweets_df.dropna(axis=0,inplace=True)
tweets_df['lons'] = tweets_df['coordinates'].map(lambda x : x.split(',')[0][1:])
tweets_df['lats'] = tweets_df['coordinates'].map(lambda x : x.split(',')[1][:-1])
tweets_df['lons'] = tweets_df['lons'].astype(float)
tweets_df['lats'] = tweets_df['lats'].astype(float)

us_map = Basemap(projection='gall',llcrnrlat=24.396308, llcrnrlon=-124.848974, 
    urcrnrlon=-66.885444,urcrnrlat=49.384358, resolution='l', area_thresh=10000)
us_map.drawcoastlines()
us_map.drawcountries()
us_map.drawstates()
us_map.drawmapboundary()
lons = list(tweets_df['lons'])
lats = list(tweets_df['lats'])
label = list(tweets_df['cluster'])
colors = ["black","red","green","blue"]
for i in range(len(lons)):
    x,y=us_map(lons[i],lats[i])
    us_map.plot(x,y,'.',color=colors[label[i]])
plt.savefig('clusters_nmf_map.png')

