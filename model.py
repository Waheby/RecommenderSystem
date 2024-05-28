import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import nltk 
from nltk.stem import PorterStemmer
import pickle

posts = pd.read_csv("test.posts.csv")
users = pd.read_csv("test.users.csv")

#DROP UNWANTED COLUMNS===============================================
users.drop('role', axis='columns', inplace=True)
users.drop('password', axis='columns', inplace=True)
users.drop('email', axis='columns', inplace=True)
users.drop('createdAt', axis='columns', inplace=True)
users.drop('_id', axis='columns', inplace=True)
users.drop('bio', axis='columns', inplace=True)
users.drop('status', axis='columns', inplace=True)
users.drop('profileImage', axis='columns', inplace=True)
users.drop('__v', axis='columns', inplace=True)

posts.drop('createdAt', axis='columns', inplace=True)
posts.drop('__v', axis='columns', inplace=True)

#COMBINE ALL COLUMNS===============================================
users['content'] = users[users.columns[1:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)

posts['content'] = posts[posts.columns[1:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)

#AGAIN DROP ALL REPEATING COLUMNS==================================
users = users[users.columns.drop(list(users.filter(regex='rating')))]
users = users[users.columns.drop(list(users.filter(regex='skills')))]
users = users[users.columns.drop(list(users.filter(regex='searches')))]
users.columns

posts = posts[posts.columns.drop(list(posts.filter(regex='tags')))]
posts.columns

posts['content'] = posts['content'].str.replace(",", " ")
users['content'] = users['content'].str.replace(",", " ")

#USE STEMMER TO REMOVE UNNECESSARY SHORT WORDS LIKE 'a, and, or, so...'
ps = PorterStemmer()

def stems(text):
    l = []
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words= 'english')

#VECTORIZE MY CONTENT COLUMN OF BOTH TABLES
vectorPosts = cv.fit_transform(posts['content']).toarray()
vectorUsers = cv.fit_transform(users['content']).toarray()

#FIND COSINE SIMILARITY ON THE VECTOR
from sklearn.metrics.pairwise import cosine_similarity

similarityPost = cosine_similarity(vectorPosts)
similarityUser = cosine_similarity(vectorUsers)

#FUNCTIONS TO TEST THE PREDICTION
def recommendpost(post):
    index = posts[posts['_id'] == post].index[0]
    distances = sorted(list(enumerate(similarityPost[index])), reverse=True, key = lambda x: x[1])
    for i in distances[1:5]:
        print(posts.iloc[i[0]].content)

def recommenduser(user):
    index = users[users['username'] == user].index[0]
    distances = sorted(list(enumerate(similarityUser[index])), reverse=True, key = lambda x: x[1])
    for i in distances[1:10]:
        print(users.iloc[i[0]].username)

recommenduser('admin1')

pickle.dump(users, open('user_list.pkl', 'wb'))
pickle.dump(similarityUser, open('similarityUser_list.pkl', 'wb'))

pickle.dump(posts, open('post_list.pkl', 'wb'))
pickle.dump(similarityPost, open('similarityPost_list.pkl', 'wb'))