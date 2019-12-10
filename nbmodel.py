import numpy as numpy
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

df = pd.read_csv('/Users/vij/Desktop/Comp562-FinalProject/winemag-data-130k-v2.csv')

# Selecting only top ten win varieties to be classified
top_ten = df['variety'].value_counts()[:10].index.tolist()
top_ten_counts = df['variety'].value_counts()[:10]
df = df[df['variety'].isin(top_ten)]

# Cleaning up dataframe to only show the relevant columns (i.e. description and variety)
cols_to_keep = ['description', 'variety']
df = df.loc[:, cols_to_keep]

df_x = df['description']
df_y = df['variety']


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2)

# print(x_test.shape, y_test.shape)

aux_stops= ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%", "flavor", "wine"]
stops = text.ENGLISH_STOP_WORDS.union(aux_stops)
count_vect = CountVectorizer(stop_words=stops)
x_train_counts = count_vect.fit_transform(x_train)


#count_vect = CountVectorizer(stop_words="english")
#x_train_counts = count_vect.fit_transform(x_train)


# making vector of counts for each word in each row in dataframe
# tfidf_vec = TfidfVectorizer(stop_words="english")
# tfidf_vec_train = tfidf_vec.fit_transform(x_train)
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
print(x_train_tfidf)
# print(tfidf_vec_train.shape)

# tfidf_vec = TfidfVectorizer(vocabulary=tfidf_vec.vocabulary_, stop_words="english")
# tfidf_vec_test = tfidf_vec.fit_transform(x_test)
# print(tfidf_vec_test.shape)

# logistRegr = LogisticRegression(multi_class="multinomial", solver='newton-cg')
# model_NB = MultinomialNB().fit(x_train_tfidf, y_train)
# model_LR = logistRegr.fit(x_train_tfidf, y_train)

# x_test_counts = count_vect.transform(x_test)
# x_test_tfidf = tfidf_transformer.transform(x_test_counts)

# predicted_varieties_NB = model_NB.predict(x_test_tfidf)
# predicted_varieties_LR = model_LR.predict(x_test_tfidf)

# print("Accuracy of Multinomial Naive Bayes: ", accuracy_score(y_test, predicted_varieties_NB))
# print("Accuracy of Logistic Regression: ", accuracy_score(y_test, predicted_varieties_LR))

# print(classification_report(y_test, predicted_varieties_NB))
# print(classification_report(y_test, predicted_varieties_LR))
