
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_similarity():
    data = pd.read_csv('./app/data/main_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l
    
# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    data = pd.read_csv('./app/data/main_data.csv')
    return list(data['movie_title'].str.capitalize())
'''
def best_movies_by_genre(genre,top_n):
    movie_score = pd.read_csv('movie_score.csv')
    return pd.DataFrame(movie_score.loc[(movie_score[genre]==1)].sort_values(['weighted_score'],ascending=False)[['title']][:top_n])
'''

def best_movies_by_genre(genre,top_n,year = 1920):
    movie_score = pd.read_csv('./app/data/movie_score.csv')
    movie_score['year'] = movie_score['title'].apply(lambda _ : int(_[-5:-1]))
    if year == '':
        year = 1920
    movie_score = movie_score[movie_score['year'] >= int(year)]
    return pd.DataFrame(movie_score.loc[(movie_score[genre]==1)].sort_values(['weighted_score'],ascending=False)[['title','count','mean','weighted_score']][:top_n])

def get_vectorizer():
    # load the nlp model and tfidf vectorizer from disk
    
    
    vectorizer = pickle.load(open('./app/models/tranform.pkl','rb'))
    return vectorizer

def getCLF():
    filename = './app/models/nlp_model.pkl'
    clf = pickle.load(open(filename, 'rb'))
    return clf


    # Book Recommender System Util functions


def loadBookData():
    popular_df = pickle.load(open('./app/models/popular.pkl','rb'))
    pt = pickle.load(open('./app/models/pt.pkl','rb'))
    books = pickle.load(open('./app/models/books.pkl','rb'))
    similarity_scores = pickle.load(open('./app/models/similarity_scores.pkl','rb'))
    return popular_df, pt, books, similarity_scores