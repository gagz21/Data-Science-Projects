# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:23:22 2019

@author: gagd2

"""

## Textmining Naive Bayes Example
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
import numpy as np

#read in the lie dtaset
reviewLie = pd.read_csv('Review_lie.csv',error_bad_lines=False)
reviewSent = pd.read_csv('Review_sentiment.csv',error_bad_lines=False)
#review the data
reviewLie.head()
reviewSent.head()
#see the column names
print(reviewLie.columns.values)
print(reviewSent.columns.values)
#see what the datatype of these columns are
reviewLie.dtypes
reviewSent.dtypes


#convert the columns into string datatype and merge all the review columns
reviewLie['Review'] = reviewLie['review'].map(str) +' '+ reviewLie['Unnamed: 2'].map(str) +' '+ reviewLie['Unnamed: 3'].map(str) +' '+ reviewLie['Unnamed: 4'].map(str) +' '+ reviewLie['Unnamed: 5'].map(str) +' '+ reviewLie['Unnamed: 6'].map(str) +' '+ reviewLie['Unnamed: 7'].map(str) +' '+ reviewLie['Unnamed: 8'].map(str) +' '+ reviewLie['Unnamed: 9'].map(str) +' '+ reviewLie['Unnamed: 10'].map(str) +' '+ reviewLie['Unnamed: 11'].map(str) +' '+ reviewLie['Unnamed: 12'].map(str) +' '+ reviewLie['Unnamed: 13'].map(str) +' '+ reviewLie['Unnamed: 14'].map(str) +' '+ reviewLie['Unnamed: 15'].map(str) +' '+ reviewLie['Unnamed: 16'].map(str) +' '+ reviewLie['Unnamed: 17'].map(str) +' '+ reviewLie['Unnamed: 18'].map(str) +' '+ reviewLie['Unnamed: 19'].map(str) +' '+ reviewLie['Unnamed: 20'].map(str) +' '+ reviewLie['Unnamed: 21'].map(str) +' '+ reviewLie['Unnamed: 22'].map(str)
reviewSent['Review'] = reviewSent['review'].map(str) +' '+ reviewSent['Unnamed: 2'].map(str) +' '+ reviewSent['Unnamed: 3'].map(str) +' '+ reviewSent['Unnamed: 4'].map(str) +' '+ reviewSent['Unnamed: 5'].map(str) +' '+ reviewSent['Unnamed: 6'].map(str) +' '+ reviewSent['Unnamed: 7'].map(str) +' '+ reviewSent['Unnamed: 8'].map(str) +' '+ reviewSent['Unnamed: 9'].map(str) +' '+ reviewSent['Unnamed: 10'].map(str) +' '+ reviewSent['Unnamed: 11'].map(str) +' '+ reviewSent['Unnamed: 12'].map(str) +' '+ reviewSent['Unnamed: 13'].map(str) +' '+ reviewSent['Unnamed: 14'].map(str) +' '+ reviewSent['Unnamed: 15'].map(str) +' '+ reviewSent['Unnamed: 16'].map(str) +' '+ reviewSent['Unnamed: 17'].map(str) +' '+ reviewSent['Unnamed: 18'].map(str) +' '+ reviewSent['Unnamed: 19'].map(str) +' '+ reviewSent['Unnamed: 20'].map(str) +' '+ reviewSent['Unnamed: 21'].map(str) +' '+ reviewSent['Unnamed: 22'].map(str)

print(reviewLie['Review'])
print(reviewSent['Review'])
#drop all the unnecessary columns and just keep the two important columns

reviewLie = reviewLie.drop(["review", "Unnamed: 2", "Unnamed: 3", "Unnamed: 4","Unnamed: 5","Unnamed: 6","Unnamed: 7","Unnamed: 8",
                            "Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13","Unnamed: 14","Unnamed: 15","Unnamed: 16",
                            "Unnamed: 17","Unnamed: 18","Unnamed: 19","Unnamed: 20","Unnamed: 21","Unnamed: 22"], axis=1)

reviewSent = reviewSent.drop(["review", "Unnamed: 2", "Unnamed: 3", "Unnamed: 4","Unnamed: 5","Unnamed: 6","Unnamed: 7","Unnamed: 8",
                              "Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13","Unnamed: 14","Unnamed: 15","Unnamed: 16",
                              "Unnamed: 17","Unnamed: 18","Unnamed: 19","Unnamed: 20","Unnamed: 21","Unnamed: 22"], axis=1)


#print(reviewLie)
#print(reviewSent)


#replace nan values
reviewLie['Review'] = reviewLie['Review'].str.replace("nan","")
reviewLie['Review'] = reviewLie['Review'].str.replace("nans","naans")
reviewLie['Review'] = reviewLie['Review'].str.replace("!","")
reviewLie['Review'] = reviewLie['Review'].str.replace("\'","")
reviewLie['Review'] = reviewLie['Review'].str.replace(".","")
reviewLie['Review'] = reviewLie['Review'].str.replace("\\","")
reviewLie['Review'] = reviewLie['Review'].str.replace(":","")
reviewLie['Review'] = reviewLie['Review'].str.replace("(","")
reviewLie['Review'] = reviewLie['Review'].str.replace(")","")


reviewSent['Review'] = reviewSent['Review'].str.replace("nan","")
reviewSent['Review'] = reviewSent['Review'].str.replace("nans","naans")
reviewSent['Review'] = reviewSent['Review'].str.replace("!","")
reviewSent['Review'] = reviewSent['Review'].str.replace("\'","")
reviewSent['Review'] = reviewSent['Review'].str.replace(".","")
reviewSent['Review'] = reviewSent['Review'].str.replace("\\","")
reviewSent['Review'] = reviewSent['Review'].str.replace(":","")
reviewSent['Review'] = reviewSent['Review'].str.replace("(","")
reviewSent['Review'] = reviewSent['Review'].str.replace(")","")


#write this into a csv file
reviewLie_file = reviewLie.to_csv (r'reviewLie_updated.csv', index = None, header=True)
reviewSent_file = reviewSent.to_csv (r'reviewSent_updated.csv', index = None, header=True)


#separate the lie label from the review
reviewLie.tail()
review = np.array(reviewLie['Review'])
lie = np.array(reviewLie['lie'])

#separate the sentiment label from the review
reviewSent.tail()
review2 = np.array(reviewSent['Review'])
sent = np.array(reviewSent['sentiment'])    

#------------------------------------Non-Binary Vectorizer for MNB Algorithm----------------------------------------------------
#----------------------------MODEL 1-----------------------------------------------------------------------------------------------
Vect=CountVectorizer(analyzer = 'word',
                        stop_words='english',
                        #token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        #tokenizer=LemmaTokenizer(),
                        #strip_accents = 'unicode', 
                        lowercase = True
                        )

X1=Vect.fit_transform(review)
ColumnNames=Vect.get_feature_names()
print("Column names: ", ColumnNames)
Vectbuilder=pd.DataFrame(X1.toarray(),columns=ColumnNames)
print(Vectbuilder)

## Now update the row names
MyDict={}
for i in range(0, len(lie)):
    MyDict[i] = lie[i]

print("MY DICT:", MyDict)
        
Vectbuilder=Vectbuilder.rename(MyDict, axis="index")
print(Vectbuilder)

## Replace the NaN with 0 because it actually 
## means none in this case
Vectbuilder=Vectbuilder.fillna(0)
print("FIRST...Normal DF Freq")  ## These print statements help you to see where you are
print(Vectbuilder)

# Build data frames to store words and TF-IDF results
words, values = pd.DataFrame(np.array(Vect.get_feature_names())), pd.DataFrame(np.array(X1.mean(axis=0))).transpose()

# Build a dataframe with the words and tf-idf results to view words with both the highest and lowest scores. 
final_results = pd.concat([words, values], axis=1)
final_results.columns = ['words', 'CountVect']
final_results = final_results.sort_values(by=['CountVect'], ascending=False)
print(final_results)
print('Top 10 highest frequency: ') 
print(final_results.iloc[:10, :], '\n')

print('Bottom 10 lowest frequency: ')
print(final_results.iloc[1245:, :])

#plot the scores on a bar plot
final_results.iloc[:10, :].plot(kind='bar',x='words', y='CountVect', title='Words with highest freq', figsize=(10,5), grid=True)
final_results.iloc[1245:, :].plot(kind='bar',x='words', y='CountVect', title='Words with lowest freq', figsize=(10,5), grid=True)

from wordcloud import WordCloud, ImageColorGenerator
# Create a dictionary with the words and TF-IDF values that will be passed into WordCloud()
d = {}
for a, x in final_results.values:
    d[a] = x

# Adjust image parameters to increase height and width of figure, create word cloud, and show image
plt.rcParams['figure.figsize'] = [10, 5]
wordcloud = WordCloud(scale=5)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

##use count vectorizer transformation on the next dataset - sentiment dataset
VectS=CountVectorizer(analyzer = 'word',
                      stop_words='english',
                      #token_pattern='(?u)[a-zA-Z]+',
                      #token_pattern=pattern,
                      #tokenizer=LemmaTokenizer(),
                      #strip_accents = 'unicode', 
                      lowercase = True
                      )

X2=VectS.fit_transform(review2)
ColumnNames2=VectS.get_feature_names()
print("Column names: ", ColumnNames2)
VectSbuilder=pd.DataFrame(X2.toarray(),columns=ColumnNames2)
print(VectSbuilder)

## Now update the row names
MyDict1={}
for i in range(0, len(sent)):
    MyDict1[i] = sent[i]

print("MY DICT:", MyDict1)
        
VectSbuilder=VectSbuilder.rename(MyDict1, axis="index")
print(VectSbuilder)

## Replace the NaN with 0 because it actually 
## means none in this case
VectSbuilder=VectSbuilder.fillna(0)
print("FIRST...Normal DF Freq")  ## These print statements help you to see where you are
print(VectSbuilder)
        

# Build data frames to store words and TF-IDF results
words, values = pd.DataFrame(np.array(VectS.get_feature_names())), pd.DataFrame(np.array(X2.mean(axis=0))).transpose()

# Build a dataframe with the words and tf-idf results to view words with both the highest and lowest scores. 
final_results = pd.concat([words, values], axis=1)
final_results.columns = ['words', 'CountVect']
final_results = final_results.sort_values(by=['CountVect'], ascending=False)
print(final_results)
print('Top 10 highest frequency: ') 
print(final_results.iloc[:10, :], '\n')

print('Bottom 10 lowest frequency: ')
print(final_results.iloc[1245:, :])

from wordcloud import WordCloud, ImageColorGenerator
# Create a dictionary with the words and TF-IDF values that will be passed into WordCloud()
d = {}
for a, x in final_results.values:
    d[a] = x

# Adjust image parameters to increase height and width of figure, create word cloud, and show image
plt.rcParams['figure.figsize'] = [10, 5]
wordcloud = WordCloud(scale=5)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#----------------------------MODEL 1-----------------------------------------------------------------------------------------------

#---------------------------Training & Test Datasets--------------------------------------------------------------------
## Create the testing set - grab a sample from the training set. 
## Be careful. Notice that right now, our train set is sorted by label.
## If your train set is large enough, you can take a random sample.
from sklearn.model_selection import train_test_split

#training and test sets for MNB algorithm model 1
Vbuilder_train, Vbuilder_test, lie_train, lie_test = train_test_split(Vectbuilder,lie, test_size=0.3)
VSbuilder_train, VSbuilder_test, sent_train, sent_test = train_test_split(VectSbuilder,sent, test_size=0.3)


####################################################################
########################### Naive Bayes ############################          MODEL 1
####################################################################
from sklearn.naive_bayes import MultinomialNB
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB.fit
#Create the modeler
MyModelNB= MultinomialNB()
NBModelLie1 = MyModelNB.fit(Vbuilder_train, lie_train)
Prediction_L1 = NBModelLie1.predict(Vbuilder_test)
print("The prediction from NB is:")
print(Prediction_L1)
print("The actual labels are:")
print(lie_test)
## confusion matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
from sklearn.metrics import confusion_matrix
labels = ['f', 't']
cnf_matrix = confusion_matrix(lie_test, Prediction_L1, labels)
print("The confusion matrix is:")
print(cnf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cnf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(NBModelLie1.predict_proba(Vbuilder_test),2))

#Accuracy score of the model
print ("Score:", NBModelLie1.score(Vbuilder_test, lie_test))


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
# Perform 10-fold cross validation
scores_L1 = cross_val_score(NBModelLie1, Vectbuilder, lie, scoring='accuracy', cv=10)
print ("Cross-validated scores:", scores_L1)
#get the mean of each fold 
print("Accuracy of Model with Cross Validation is:",scores_L1.mean() * 100)

# Make cross validated predictions
predictions_cvL1 = cross_val_predict(NBModelLie1, Vectbuilder, lie, cv=10)
plt.scatter(lie, predictions_cvL1)



#---------------------------Applying NB model on sentiment dataset---------------------------------------------------
NBModelSent1 = MyModelNB.fit(VSbuilder_train, sent_train)
Prediction_S1 = NBModelSent1.predict(VSbuilder_test)
print("The prediction from NB is:")
print(Prediction_S1)
print("The actual labels are:")
print(sent_test)

## confusion matrix
labels = ['p', 'n']
cnf_matrix = confusion_matrix(sent_test, Prediction_S1, labels)
print("The confusion matrix is:")
print(cnf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cnf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
### prediction probabilities
print(np.round(NBModelSent1.predict_proba(VSbuilder_test),2))
#Accuracy score of the model
print ("Score:", NBModelSent1.score(VSbuilder_test, sent_test))

scores_S1 = cross_val_score(NBModelSent1, VectSbuilder, sent, scoring='accuracy', cv=10)
print ("Cross-validated scores:", scores_S1)

#get the mean of each fold 
print("Accuracy of Model with Cross Validation is:",scores_S1.mean() * 100)

predictions_cvS1 = cross_val_predict(NBModelSent1, VectSbuilder, sent, cv=10)
plt.scatter(sent, predictions_cvS1)


#----------------------------MODEL 2--------ONLY FOR LIE LABEL DATASET---------------------------------------------------------------------------------------

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles) if t not in stopwords.words('english') or string.punctuation]
    


##Build the vectorizer
#pattern='r/^[a-zA-Z]{4}$/'
#pattern="[^r\P{P}]+"
  
Vect2=CountVectorizer(analyzer = 'word',
                     stop_words='english',
                     #token_pattern='(?u)[a-zA-Z]+',
                     #token_pattern=pattern,
                     tokenizer=LemmaTokenizer(),
                     #strip_accents = 'unicode', 
                     lowercase = True
                     )

X1_2=Vect2.fit_transform(review)
ColumnNames=Vect2.get_feature_names()
print("Column names: ", ColumnNames)
Vectbuilder2=pd.DataFrame(X1_2.toarray(),columns=ColumnNames)
print(Vectbuilder2)

## Now update the row names
MyDict={}
for i in range(0, len(lie)):
    MyDict[i] = lie[i]

print("MY DICT:", MyDict)
        
Vectbuilder2=Vectbuilder2.rename(MyDict, axis="index")
print(Vectbuilder2)

## Replace the NaN with 0 because it actually 
## means none in this case
Vectbuilder2=Vectbuilder2.fillna(0)
print("FIRST...Normal DF Freq")  ## These print statements help you to see where you are
print(Vectbuilder2)

# Build data frames to store words and TF-IDF results
words, values = pd.DataFrame(np.array(Vect2.get_feature_names())), pd.DataFrame(np.array(X1_2.mean(axis=0))).transpose()

# Build a dataframe with the words and tf-idf results to view words with both the highest and lowest scores. 
final_results = pd.concat([words, values], axis=1)
final_results.columns = ['words', 'CountVect']
final_results = final_results.sort_values(by=['CountVect'], ascending=False)
print(final_results)
print('Top 10 highest frequency: ') 
print(final_results.iloc[:10, :], '\n')

print('Bottom 10 lowest frequency: ')
print(final_results.iloc[1270:, :])

from wordcloud import WordCloud, ImageColorGenerator
# Create a dictionary with the words and TF-IDF values that will be passed into WordCloud()
d = {}
for a, x in final_results.values:
    d[a] = x

# Adjust image parameters to increase height and width of figure, create word cloud, and show image
plt.rcParams['figure.figsize'] = [10, 5]
wordcloud = WordCloud(scale=5)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#----------------------------MODEL 2--------ONLY FOR SENTIMENT LABEL DATASET---------------------------------------------------------------------------------------
Vect2S=CountVectorizer(analyzer = 'word',
                      stop_words='english',
                      #token_pattern='(?u)[a-zA-Z]+',
                      #token_pattern=pattern,
                      tokenizer=LemmaTokenizer(),
                      #strip_accents = 'unicode', 
                      lowercase = True
                      )

X2_1=Vect2S.fit_transform(review2)
ColumnNames2=Vect2S.get_feature_names()
print("Column names: ", ColumnNames2)
Vect2Sbuilder=pd.DataFrame(X2_1.toarray(),columns=ColumnNames2)
print(Vect2Sbuilder)

## Now update the row names
MyDict1={}
for i in range(0, len(sent)):
    MyDict1[i] = sent[i]

print("MY DICT:", MyDict1)
        
Vect2Sbuilder=Vect2Sbuilder.rename(MyDict1, axis="index")
print(Vect2Sbuilder)

## Replace the NaN with 0 because it actually 
## means none in this case
Vect2Sbuilder=Vect2Sbuilder.fillna(0)
print("FIRST...Normal DF Freq")  ## These print statements help you to see where you are
print(Vect2Sbuilder)
        

# Build data frames to store words and TF-IDF results
words, values = pd.DataFrame(np.array(Vect2S.get_feature_names())), pd.DataFrame(np.array(X2_1.mean(axis=0))).transpose()

# Build a dataframe with the words and tf-idf results to view words with both the highest and lowest scores. 
final_results = pd.concat([words, values], axis=1)
final_results.columns = ['words', 'CountVect']
final_results = final_results.sort_values(by=['CountVect'], ascending=False)
print(final_results)
print('Top 10 highest frequency: ') 
print(final_results.iloc[:10, :], '\n')

print('Bottom 10 lowest frequency: ')
print(final_results.iloc[1272:, :])


from wordcloud import WordCloud, ImageColorGenerator
# Create a dictionary with the words and TF-IDF values that will be passed into WordCloud()
d = {}
for a, x in final_results.values:
    d[a] = x

# Adjust image parameters to increase height and width of figure, create word cloud, and show image
plt.rcParams['figure.figsize'] = [10, 5]
wordcloud = WordCloud(scale=5)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#----------------------------MODEL 2-----------------------------------------------------------------------------------------------


#training and test sets for MNB algorithm model 2
Vbuilder2_train, Vbuilder2_test, lie_train, lie_test = train_test_split(Vectbuilder2,lie, test_size=0.3)
VSbuilder2_train, VSbuilder2_test, sent_train, sent_test = train_test_split(Vect2Sbuilder,sent, test_size=0.3)




####################################################################
########################### Naive Bayes ############################          MODEL 2
####################################################################
from sklearn.naive_bayes import MultinomialNB
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB.fit
#Create the modeler
MyModelNB= MultinomialNB()
NBModelLie2 = MyModelNB.fit(Vbuilder2_train, lie_train)
Prediction_L2 = NBModelLie2.predict(Vbuilder2_test)
print("The prediction from NB is:")
print(Prediction_L2)
print("The actual labels are:")
print(lie_test)
## confusion matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
from sklearn.metrics import confusion_matrix
labels = ['f', 't']
cnf_matrix = confusion_matrix(lie_test, Prediction_L2, labels)
print("The confusion matrix is:")
print(cnf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cnf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(NBModelLie2.predict_proba(Vbuilder2_test),2))

#Accuracy score of the model
print ("Score:", NBModelLie2.score(Vbuilder2_test, lie_test))


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
# Perform 10-fold cross validation
scores_L2 = cross_val_score(NBModelLie2, Vectbuilder2, lie, scoring='accuracy', cv=10)
print ("Cross-validated scores:", scores_L2)
#get the mean of each fold 
print("Accuracy of Model with Cross Validation is:",scores_L2.mean() * 100)

# Make cross validated predictions
predictions_cvL2 = cross_val_predict(NBModelLie2, Vectbuilder2, lie, cv=10)
plt.scatter(lie, predictions_cvL2)



#----------------------------------------Applying NB model on sentiment dataset-------------------------------------

NBModelSent2 = MyModelNB.fit(VSbuilder2_train, sent_train)
Prediction_S2 = NBModelSent2.predict(VSbuilder2_test)
print("The prediction from NB is:")
print(Prediction_S2)
print("The actual labels are:")
print(sent_test)

## confusion matrix
labels = ['p', 'n']
cnf_matrix = confusion_matrix(sent_test, Prediction_S2, labels)
print("The confusion matrix is:")
print(cnf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cnf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
### prediction probabilities
print(np.round(NBModelSent2.predict_proba(VSbuilder2_test),2))
#Accuracy score of the model
print ("Score:", NBModelSent2.score(VSbuilder2_test, sent_test))

scores_S2 = cross_val_score(NBModelSent2, Vect2Sbuilder, sent, scoring='accuracy', cv=10)
print ("Cross-validated scores:", scores_S2)

#get the mean of each fold 
print("Accuracy of Model with Cross Validation is:",scores_S2.mean() * 100)

predictions_cvS2 = cross_val_predict(NBModelSent2, Vect2Sbuilder, sent, cv=10)
plt.scatter(sent, predictions_cvS2)




#----------------------------MODEL 3--------ONLY FOR LIE LABEL DATASET---------------------------------------------------------------------------------------

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles) if t not in stopwords.words('english') or string.punctuation]
    


##Build the vectorizer
pattern='r/^[a-zA-Z]{4}$/'
#pattern="[^r\P{P}]+"
  
Vect3=CountVectorizer(analyzer = 'word',
                     stop_words='english',
                     #token_pattern='(?u)[a-zA-Z]+',
                     token_pattern=pattern,
                     tokenizer=LemmaTokenizer(),
                     #strip_accents = 'unicode', 
                     lowercase = True
                     )

X1_3=Vect3.fit_transform(review)
ColumnNames=Vect3.get_feature_names()
print("Column names: ", ColumnNames)
Vectbuilder3=pd.DataFrame(X1_3.toarray(),columns=ColumnNames)
print(Vectbuilder3)

## Now update the row names
MyDict={}
for i in range(0, len(lie)):
    MyDict[i] = lie[i]

print("MY DICT:", MyDict)
        
Vectbuilder3=Vectbuilder3.rename(MyDict, axis="index")
print(Vectbuilder3)

## Replace the NaN with 0 because it actually 
## means none in this case
Vectbuilder3=Vectbuilder3.fillna(0)
print("FIRST...Normal DF Freq")  ## These print statements help you to see where you are
print(Vectbuilder3)

    
# Build data frames to store words and TF-IDF results
words, values = pd.DataFrame(np.array(Vect3.get_feature_names())), pd.DataFrame(np.array(X1_3.mean(axis=0))).transpose()

# Build a dataframe with the words and tf-idf results to view words with both the highest and lowest scores. 
final_results = pd.concat([words, values], axis=1)
final_results.columns = ['words', 'CountVect']
final_results = final_results.sort_values(by=['CountVect'], ascending=False)
print(final_results)
print('Top 10 highest frequency: ') 
print(final_results.iloc[:10, :], '\n')

print('Bottom 10 lowest frequency: ')
print(final_results.iloc[1270:, :])

from wordcloud import WordCloud, ImageColorGenerator
# Create a dictionary with the words and TF-IDF values that will be passed into WordCloud()
d = {}
for a, x in final_results.values:
    d[a] = x

# Adjust image parameters to increase height and width of figure, create word cloud, and show image
plt.rcParams['figure.figsize'] = [10, 5]
wordcloud = WordCloud(scale=5)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#----------------------------MODEL 3--------ONLY FOR SENTIMENT LABEL DATASET---------------------------------------------------------------------------------------
Vect3S=CountVectorizer(analyzer = 'word',
                      stop_words='english',
                      token_pattern='(?u)[a-zA-Z]+',
                      #token_pattern=pattern,
                      tokenizer=LemmaTokenizer(),
                      #strip_accents = 'unicode', 
                      lowercase = True
                      )

X2_3=Vect3S.fit_transform(review2)
ColumnNames2=Vect3S.get_feature_names()
print("Column names: ", ColumnNames2)
Vect3Sbuilder=pd.DataFrame(X2_3.toarray(),columns=ColumnNames2)
print(Vect3Sbuilder)

## Now update the row names
MyDict1={}
for i in range(0, len(sent)):
    MyDict1[i] = sent[i]

print("MY DICT:", MyDict1)
        
Vect3Sbuilder=Vect3Sbuilder.rename(MyDict1, axis="index")
print(Vect3Sbuilder)

## Replace the NaN with 0 because it actually 
## means none in this case

Vect3Sbuilder=Vect3Sbuilder.fillna(0)
print("FIRST...Normal DF Freq")  ## These print statements help you to see where you are
print(Vect3Sbuilder)
        

# Build data frames to store words and TF-IDF results
words, values = pd.DataFrame(np.array(Vect3S.get_feature_names())), pd.DataFrame(np.array(X2_3.mean(axis=0))).transpose()

# Build a dataframe with the words and tf-idf results to view words with both the highest and lowest scores. 
final_results = pd.concat([words, values], axis=1)
final_results.columns = ['words', 'CountVect']
final_results = final_results.sort_values(by=['CountVect'], ascending=False)
print(final_results)
print('Top 10 highest frequency: ') 
print(final_results.iloc[:10, :], '\n')

print('Bottom 10 lowest frequency: ')
print(final_results.iloc[1272:, :])


from wordcloud import WordCloud, ImageColorGenerator
# Create a dictionary with the words and TF-IDF values that will be passed into WordCloud()
d = {}
for a, x in final_results.values:
    d[a] = x

# Adjust image parameters to increase height and width of figure, create word cloud, and show image
plt.rcParams['figure.figsize'] = [10, 5]
wordcloud = WordCloud(scale=5)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#----------------------------MODEL 3-----------------------------------------------------------------------------------------------


#training and test sets for MNB algorithm model 3
Vbuilder3_train, Vbuilder3_test, lie_train, lie_test = train_test_split(Vectbuilder3,lie, test_size=0.3)
VSbuilder3_train, VSbuilder3_test, sent_train, sent_test = train_test_split(Vect3Sbuilder,sent, test_size=0.3)


####################################################################
########################### Naive Bayes ############################          MODEL 3
####################################################################
from sklearn.naive_bayes import MultinomialNB
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB.fit
#Create the modeler
MyModelNB= MultinomialNB()
NBModelLie3 = MyModelNB.fit(Vbuilder3_train, lie_train)
Prediction_L3 = NBModelLie3.predict(Vbuilder3_test)
print("The prediction from NB is:")
print(Prediction_L3)
print("The actual labels are:")
print(lie_test)
## confusion matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
from sklearn.metrics import confusion_matrix
labels = ['f', 't']
cnf_matrix = confusion_matrix(lie_test, Prediction_L3, labels)
print("The confusion matrix is:")
print(cnf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cnf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(NBModelLie3.predict_proba(Vbuilder3_test),2))

#Accuracy score of the model
print ("Score:", NBModelLie3.score(Vbuilder3_test, lie_test))


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
# Perform 10-fold cross validation
scores_L3 = cross_val_score(NBModelLie3, Vectbuilder3, lie, scoring='accuracy', cv=10)
print ("Cross-validated scores:", scores_L3)
#get the mean of each fold 
print("Accuracy of Model with Cross Validation is:",scores_L3.mean() * 100)

# Make cross validated predictions
predictions_cvL3 = cross_val_predict(NBModelLie3, Vectbuilder3, lie, cv=10)
plt.scatter(lie, predictions_cvL3)



#-----------------------------------Applying NB model on sentiment dataset----------------------------------------------
NBModelSent3 = MyModelNB.fit(VSbuilder3_train, sent_train)
Prediction_S3 = NBModelSent3.predict(VSbuilder3_test)
print("The prediction from NB is:")
print(Prediction_S3)
print("The actual labels are:")
print(sent_test)

## confusion matrix
labels = ['p', 'n']
cnf_matrix = confusion_matrix(sent_test, Prediction_S3, labels)
print("The confusion matrix is:")
print(cnf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cnf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
### prediction probabilities
print(np.round(NBModelSent3.predict_proba(VSbuilder3_test),2))
#Accuracy score of the model
print ("Score:", NBModelSent3.score(VSbuilder3_test, sent_test))

scores_S3 = cross_val_score(NBModelSent3, Vect3Sbuilder, sent, scoring='accuracy', cv=10)
print ("Cross-validated scores:", scores_S3)

#get the mean of each fold 
print("Accuracy of Model with Cross Validation is:",scores_S3.mean() * 100)

predictions_cvS3 = cross_val_predict(NBModelSent3, Vect3Sbuilder, sent, cv=10)
plt.scatter(sent, predictions_cvS3)



####################################################################
########################### Naive Bayes ############################          MODEL 3
####################################################################




#------------------------------------Non-Binary Vectorizer for MNB Algorithm----------------------------------------------------





#------------------------------------Binary Vectorizer for Bernoulli Algorithm----------------------------------------------------
#----------------------------MODEL 1-----------------------------------------------------------------------------------------------

BVect=CountVectorizer(analyzer = 'word',
                        stop_words='english',
                        #token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        #tokenizer=LemmaTokenizer(),
                        #strip_accents = 'unicode', 
                        lowercase = True,
                        binary=True
                        )

X3=BVect.fit_transform(review)
ColumnNames3=BVect.get_feature_names()
print("Column names: ", ColumnNames3)
BVectbuilder=pd.DataFrame(X3.toarray(),columns=ColumnNames3)
print(BVectbuilder)

## Now update the row names
MyDict={}
for i in range(0, len(lie)):
    MyDict[i] = lie[i]

print("MY DICT:", MyDict)
        
BVectbuilder=BVectbuilder.rename(MyDict, axis="index")
print(BVectbuilder)

## Replace the NaN with 0 because it actually 
## means none in this case
BVectbuilder=BVectbuilder.fillna(0)
print("FIRST...Normal DF Freq")  ## These print statements help you to see where you are
print(BVectbuilder)

# Build data frames to store words and TF-IDF results
words, values = pd.DataFrame(np.array(BVect.get_feature_names())), pd.DataFrame(np.array(X3.mean(axis=0))).transpose()

# Build a dataframe with the words and tf-idf results to view words with both the highest and lowest scores. 
final_results = pd.concat([words, values], axis=1)
final_results.columns = ['words', 'CountVect']
final_results = final_results.sort_values(by=['CountVect'], ascending=False)
print(final_results)
print('Top 10 highest frequency: ') 
print(final_results.iloc[:10, :], '\n')

print('Bottom 10 lowest frequency: ')
print(final_results.iloc[1245:, :])

from wordcloud import WordCloud, ImageColorGenerator
# Create a dictionary with the words and TF-IDF values that will be passed into WordCloud()
d = {}
for a, x in final_results.values:
    d[a] = x

# Adjust image parameters to increase height and width of figure, create word cloud, and show image
plt.rcParams['figure.figsize'] = [10, 5]
wordcloud = WordCloud(scale=5)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

##use count vectorizer transformation on the next dataset - sentiment dataset
BVectS=CountVectorizer(analyzer = 'word',
                        stop_words='english',
                        #token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        #tokenizer=LemmaTokenizer(),
                        #strip_accents = 'unicode', 
                        lowercase = True,
                        binary=True
                        )

X4=BVectS.fit_transform(review2)
ColumnNames4=BVectS.get_feature_names()
print("Column names: ", ColumnNames4)
BVectSbuilder=pd.DataFrame(X4.toarray(),columns=ColumnNames4)
print(BVectSbuilder)

## Now update the row names
MyDict1={}
for i in range(0, len(sent)):
    MyDict1[i] = sent[i]

print("MY DICT:", MyDict1)
        
BVectSbuilder=BVectSbuilder.rename(MyDict1, axis="index")
print(BVectSbuilder)

## Replace the NaN with 0 because it actually 
## means none in this case
BVectSbuilder=BVectSbuilder.fillna(0)
print("FIRST...Normal DF Freq")  ## These print statements help you to see where you are
print(BVectSbuilder)
        

# Build data frames to store words and TF-IDF results
words, values = pd.DataFrame(np.array(BVectS.get_feature_names())), pd.DataFrame(np.array(X4.mean(axis=0))).transpose()

# Build a dataframe with the words and tf-idf results to view words with both the highest and lowest scores. 
final_results = pd.concat([words, values], axis=1)
final_results.columns = ['words', 'CountVect']
final_results = final_results.sort_values(by=['CountVect'], ascending=False)
print(final_results)
print('Top 10 highest frequency: ') 
print(final_results.iloc[:10, :], '\n')

print('Bottom 10 lowest frequency: ')
print(final_results.iloc[1245:, :])

from wordcloud import WordCloud, ImageColorGenerator
# Create a dictionary with the words and TF-IDF values that will be passed into WordCloud()
d = {}
for a, x in final_results.values:
    d[a] = x

# Adjust image parameters to increase height and width of figure, create word cloud, and show image
plt.rcParams['figure.figsize'] = [10, 5]
wordcloud = WordCloud(scale=5)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#----------------------------MODEL 1-----------------------------------------------------------------------------------------------


#training and test sets for Bernoulli NB algorithm model 1
BVbuilder_train, BVbuilder_test, lie_train, lie_test = train_test_split(BVectbuilder,lie, test_size=0.3)
BVSbuilder_train, BVSbuilder_test, sent_train, sent_test = train_test_split(BVectSbuilder,sent, test_size=0.3)


#######################################################
### Bernoulli #########################################          MODEL 1
#######################################################

### Lie Dataset--------------------------------------------------
from sklearn.naive_bayes import BernoulliNB
BernModelL1 = BernoulliNB()
BernModelL1.fit(BVbuilder_train, lie_train)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Prediction_BL1 = BernModelL1.predict(BVbuilder_test)
print("Bernoulli prediction:\n", Prediction_BL1)
print("Actual:")
print(lie_test)

## confusion matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
from sklearn.metrics import confusion_matrix
labels = ['f', 't']
cnf_matrix = confusion_matrix(lie_test, Prediction_BL1, labels)
print("The confusion matrix is:")
print(cnf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cnf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(BernModelL1.predict_proba(BVbuilder_test),2))

#Accuracy score of the model
print ("Score:", BernModelL1.score(BVbuilder_test, lie_test))

#cross-validation method
scores_BL1 = cross_val_score(BernModelL1, BVectbuilder, lie, scoring='accuracy', cv=10)
print ("Cross-validated scores:", scores_BL1)

#get the mean of each fold 
print("Accuracy of Model with Cross Validation is:",scores_BL1.mean() * 100)

predictions_cvBL1 = cross_val_predict(BernModelL1, BVectbuilder, lie, cv=10)
plt.scatter(lie, predictions_cvBL1)

### Lie Dataset--------------------------------------------------


### Sent Dataset--------------------------------------------------

BernModelSentS1 = BernoulliNB()
BernModelSentS1.fit(BVSbuilder_train, sent_train)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Prediction_BS1 = BernModelSentS1.predict(BVSbuilder_test)
print("Bernoulli prediction:\n", Prediction_BS1)
print("Actual:")
print(sent_test)

## confusion matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
from sklearn.metrics import confusion_matrix
labels = ['n', 'p']
cnf_matrix = confusion_matrix(sent_test, Prediction_BS1, labels)
print("The confusion matrix is:")
print(cnf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cnf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(BernModelSentS1.predict_proba(BVSbuilder_test),2))

#Accuracy score of the model
print ("Score:", BernModelSentS1.score(BVSbuilder_test, sent_test))

#cross-validation method
scores_BS1 = cross_val_score(BernModelSentS1, BVectSbuilder, sent, scoring='accuracy', cv=10)
print ("Cross-validated scores:", scores_BS1)

#get the mean of each fold 
print("Accuracy of Model with Cross Validation is:",scores_BS1.mean() * 100)

predictions_cvBS1 = cross_val_predict(BernModelSentS1, BVectSbuilder, sent, cv=10)
plt.scatter(sent, predictions_cvBS1)

### Sent Dataset--------------------------------------------------


#----------------------------MODEL 2--------ONLY FOR LIE LABEL DATASET---------------------------------------------------------------------------------------

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles) if t not in stopwords.words('english') or string.punctuation]
    


##Build the vectorizer
#pattern='r/^[a-zA-Z]{4}$/'
#pattern="[^r\P{P}]+"
  
BVect2=CountVectorizer(analyzer = 'word',
                     stop_words='english',
                     #token_pattern='(?u)[a-zA-Z]+',
                     #token_pattern=pattern,
                     tokenizer=LemmaTokenizer(),
                     #strip_accents = 'unicode', 
                     lowercase = True,
                     binary = True
                     )

X3_1=BVect2.fit_transform(review)
ColumnNames=BVect2.get_feature_names()
print("Column names: ", ColumnNames)
BVectbuilder2=pd.DataFrame(X3_1.toarray(),columns=ColumnNames)
print(BVectbuilder2)

## Now update the row names
MyDict={}
for i in range(0, len(lie)):
    MyDict[i] = lie[i]

print("MY DICT:", MyDict)
        
BVectbuilder2=BVectbuilder2.rename(MyDict, axis="index")
print(BVectbuilder2)

## Replace the NaN with 0 because it actually 
## means none in this case
BVectbuilder2=BVectbuilder2.fillna(0)
print("FIRST...Normal DF Freq")  ## These print statements help you to see where you are
print(BVectbuilder2)

# Build data frames to store words and TF-IDF results
words, values = pd.DataFrame(np.array(BVect2.get_feature_names())), pd.DataFrame(np.array(X3_1.mean(axis=0))).transpose()

# Build a dataframe with the words and tf-idf results to view words with both the highest and lowest scores. 
final_results = pd.concat([words, values], axis=1)
final_results.columns = ['words', 'CountVect']
final_results = final_results.sort_values(by=['CountVect'], ascending=False)
print(final_results)
print('Top 10 highest frequency: ') 
print(final_results.iloc[:10, :], '\n')

print('Bottom 10 lowest frequency: ')
print(final_results.iloc[1270:, :])

from wordcloud import WordCloud, ImageColorGenerator
# Create a dictionary with the words and TF-IDF values that will be passed into WordCloud()
d = {}
for a, x in final_results.values:
    d[a] = x

# Adjust image parameters to increase height and width of figure, create word cloud, and show image
plt.rcParams['figure.figsize'] = [10, 5]
wordcloud = WordCloud(scale=5)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#----------------------------MODEL 2--------ONLY FOR SENTIMENT LABEL DATASET---------------------------------------------------------------------------------------
BVectS2=CountVectorizer(analyzer = 'word',
                      stop_words='english',
                      #token_pattern='(?u)[a-zA-Z]+',
                      #token_pattern=pattern,
                      tokenizer=LemmaTokenizer(),
                      #strip_accents = 'unicode', 
                      lowercase = True,
                      binary = True
                      )

X4_1=BVectS2.fit_transform(review2)
ColumnNames2=BVectS2.get_feature_names()
print("Column names: ", ColumnNames2)
BVectS2builder=pd.DataFrame(X4_1.toarray(),columns=ColumnNames2)
print(BVectS2builder)

## Now update the row names
MyDict1={}
for i in range(0, len(sent)):
    MyDict1[i] = sent[i]

print("MY DICT:", MyDict1)
        
BVectS2builder=BVectS2builder.rename(MyDict1, axis="index")
print(BVectS2builder)

## Replace the NaN with 0 because it actually 
## means none in this case
BVectS2builder=BVectS2builder.fillna(0)
print("FIRST...Normal DF Freq")  ## These print statements help you to see where you are
print(BVectS2builder)
        

# Build data frames to store words and TF-IDF results
words, values = pd.DataFrame(np.array(BVectS2.get_feature_names())), pd.DataFrame(np.array(X4_1.mean(axis=0))).transpose()

# Build a dataframe with the words and tf-idf results to view words with both the highest and lowest scores. 
final_results = pd.concat([words, values], axis=1)
final_results.columns = ['words', 'CountVect']
final_results = final_results.sort_values(by=['CountVect'], ascending=False)
print(final_results)
print('Top 10 highest frequency: ') 
print(final_results.iloc[:10, :], '\n')

print('Bottom 10 lowest frequency: ')
print(final_results.iloc[1272:, :])

from wordcloud import WordCloud, ImageColorGenerator
# Create a dictionary with the words and TF-IDF values that will be passed into WordCloud()
d = {}
for a, x in final_results.values:
    d[a] = x

# Adjust image parameters to increase height and width of figure, create word cloud, and show image
plt.rcParams['figure.figsize'] = [10, 5]
wordcloud = WordCloud(scale=5)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
#----------------------------MODEL 2-----------------------------------------------------------------------------------------------

#training and test sets for Bernoulli NB algorithm model 2
BVbuilder2_train, BVbuilder2_test, lie_train, lie_test = train_test_split(BVectbuilder2,lie, test_size=0.3)
BVSbuilder2_train, BVSbuilder2_test, sent_train, sent_test = train_test_split(BVectS2builder,sent, test_size=0.3)


#######################################################
### Bernoulli #########################################          MODEL 2
#######################################################
### Lie Dataset--------------------------------------------------
from sklearn.naive_bayes import BernoulliNB
BernModelL2 = BernoulliNB()
BernModelL2.fit(BVbuilder2_train, lie_train)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Prediction_BL2 = BernModelL2.predict(BVbuilder2_test)
print("Bernoulli prediction:\n", Prediction_BL2)
print("Actual:")
print(lie_test)

## confusion matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
from sklearn.metrics import confusion_matrix
labels = ['f', 't']
cnf_matrix = confusion_matrix(lie_test, Prediction_BL2, labels)
print("The confusion matrix is:")
print(cnf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cnf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(BernModelL2.predict_proba(BVbuilder2_test),2))

#Accuracy score of the model
print ("Score:", BernModelL2.score(BVbuilder2_test, lie_test))

#cross-validation method
scores_BL2 = cross_val_score(BernModelL2, BVectbuilder2, lie, scoring='accuracy', cv=10)
print ("Cross-validated scores:", scores_BL2)

#get the mean of each fold 
print("Accuracy of Model with Cross Validation is:",scores_BL2.mean() * 100)

predictions_cvBL2 = cross_val_predict(BernModelL2, BVectbuilder2, lie,  cv=10)
plt.scatter(lie, predictions_cvBL2)

### Lie Dataset--------------------------------------------------


### Sent Dataset--------------------------------------------------

BernModelSentS2 = BernoulliNB()
BernModelSentS2.fit(BVSbuilder2_train, sent_train)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Prediction_BS2 = BernModelSentS2.predict(BVSbuilder2_test)
print("Bernoulli prediction:\n", Prediction_BS2)
print("Actual:")
print(sent_test)

## confusion matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
from sklearn.metrics import confusion_matrix
labels = ['n', 'p']
cnf_matrix = confusion_matrix(sent_test, Prediction_BS2, labels)
print("The confusion matrix is:")
print(cnf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cnf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(BernModelSentS2.predict_proba(BVSbuilder2_test),2))

#Accuracy score of the model
print ("Score:", BernModelSentS2.score(BVSbuilder2_test, sent_test))

#cross-validation method
scores_BS2 = cross_val_score(BernModelSentS2, BVectS2builder, sent, scoring='accuracy',  cv=10)
print ("Cross-validated scores:", scores_BS2)

#get the mean of each fold 
print("Accuracy of Model with Cross Validation is:",scores_BS2.mean() * 100)

predictions_cvBS2 = cross_val_predict(BernModelSentS2, BVectS2builder, sent, cv=10)
plt.scatter(sent, predictions_cvBS2)

### Sent Dataset--------------------------------------------------




#----------------------------MODEL 3--------ONLY FOR LIE LABEL DATASET---------------------------------------------------------------------------------------

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles) if t not in stopwords.words('english') or string.punctuation]
    


##Build the vectorizer
pattern='r/^[a-zA-Z]{4}$/'
#pattern="[^r\P{P}]+"
  
BVect3=CountVectorizer(analyzer = 'word',
                     stop_words='english',
                     token_pattern='(?u)[a-zA-Z]+',
                     #token_pattern=pattern,
                     tokenizer=LemmaTokenizer(),
                     #strip_accents = 'unicode', 
                     lowercase = True,
                     binary = True
                     )

X3_2=BVect3.fit_transform(review)
ColumnNames=BVect3.get_feature_names()
print("Column names: ", ColumnNames)
BVectbuilder3=pd.DataFrame(X3_2.toarray(),columns=ColumnNames)
print(BVectbuilder3)

## Now update the row names
MyDict={}
for i in range(0, len(lie)):
    MyDict[i] = lie[i]

print("MY DICT:", MyDict)
        
BVectbuilder3=BVectbuilder3.rename(MyDict, axis="index")
print(BVectbuilder3)

## Replace the NaN with 0 because it actually 
## means none in this case
BVectbuilder3=BVectbuilder3.fillna(0)
print("FIRST...Normal DF Freq")  ## These print statements help you to see where you are
print(BVectbuilder3)

# Build data frames to store words and TF-IDF results
words, values = pd.DataFrame(np.array(BVect3.get_feature_names())), pd.DataFrame(np.array(X3_2.mean(axis=0))).transpose()

# Build a dataframe with the words and tf-idf results to view words with both the highest and lowest scores. 
final_results = pd.concat([words, values], axis=1)
final_results.columns = ['words', 'CountVect']
final_results = final_results.sort_values(by=['CountVect'], ascending=False)
print(final_results)
print('Top 10 highest frequency: ') 
print(final_results.iloc[:10, :], '\n')

print('Bottom 10 lowest frequency: ')
print(final_results.iloc[1270:, :])

from wordcloud import WordCloud, ImageColorGenerator
# Create a dictionary with the words and TF-IDF values that will be passed into WordCloud()
d = {}
for a, x in final_results.values:
    d[a] = x

# Adjust image parameters to increase height and width of figure, create word cloud, and show image
plt.rcParams['figure.figsize'] = [10, 5]
wordcloud = WordCloud(scale=5)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#----------------------------MODEL 3--------ONLY FOR SENTIMENT LABEL DATASET---------------------------------------------------------------------------------------
BVectS3=CountVectorizer(analyzer = 'word',
                      stop_words='english',
                      token_pattern='(?u)[a-zA-Z]+',
                      #token_pattern=pattern,
                      tokenizer=LemmaTokenizer(),
                      #strip_accents = 'unicode', 
                      lowercase = True,
                      binary = True
                      )

X4_2=BVectS3.fit_transform(review2)
ColumnNames2=BVectS3.get_feature_names()
print("Column names: ", ColumnNames2)
BVectS3builder=pd.DataFrame(X4_2.toarray(),columns=ColumnNames2)
print(BVectS3builder)

## Now update the row names
MyDict1={}
for i in range(0, len(sent)):
    MyDict1[i] = sent[i]

print("MY DICT:", MyDict1)
        
BVectS3builder=BVectS3builder.rename(MyDict1, axis="index")
print(BVectS3builder)

## Replace the NaN with 0 because it actually 
## means none in this case
BVectS3builder=BVectS3builder.fillna(0)
print("FIRST...Normal DF Freq")  ## These print statements help you to see where you are
print(BVectS3builder)
        

# Build data frames to store words and TF-IDF results
words, values = pd.DataFrame(np.array(BVectS3.get_feature_names())), pd.DataFrame(np.array(X4_2.mean(axis=0))).transpose()

# Build a dataframe with the words and tf-idf results to view words with both the highest and lowest scores. 
final_results = pd.concat([words, values], axis=1)
final_results.columns = ['words', 'CountVect']
final_results = final_results.sort_values(by=['CountVect'], ascending=False)
print(final_results)
print('Top 10 highest frequency: ') 
print(final_results.iloc[:10, :], '\n')

print('Bottom 10 lowest frequency: ')
print(final_results.iloc[1272:, :])

from wordcloud import WordCloud, ImageColorGenerator
# Create a dictionary with the words and TF-IDF values that will be passed into WordCloud()
d = {}
for a, x in final_results.values:
    d[a] = x

# Adjust image parameters to increase height and width of figure, create word cloud, and show image
plt.rcParams['figure.figsize'] = [10, 5]
wordcloud = WordCloud(scale=5)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
#----------------------------MODEL 3-----------------------------------------------------------------------------------------------

#------------------------------------Binary Vectorizer for Bernoulli Algorithm----------------------------------------------------


#---------------------------Training & Test Datasets--------------------------------------------------------------------
## Create the testing set - grab a sample from the training set. 
## Be careful. Notice that right now, our train set is sorted by label.
## If your train set is large enough, you can take a random sample.

#training and test sets for Bernoulli NB algorithm model 3
BVbuilder3_train, BVbuilder3_test, lie_train, lie_test = train_test_split(BVectbuilder3,lie, test_size=0.3)
BVSbuilder3_train, BVSbuilder3_test, sent_train, sent_test = train_test_split(BVectS3builder,sent, test_size=0.3)

#---------------------------Training & Test Datasets--------------------------------------------------------------------


#-----------------------------Application of Algorithms------------------------------------------------------------------


#######################################################
### Bernoulli #########################################          MODEL 3
#######################################################
### Lie Dataset--------------------------------------------------
from sklearn.naive_bayes import BernoulliNB
BernModelL3 = BernoulliNB()
BernModelL3.fit(BVbuilder3_train, lie_train)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Prediction_BL3 = BernModelL3.predict(BVbuilder3_test)
print("Bernoulli prediction:\n", Prediction_BL3)
print("Actual:")
print(lie_test)

## confusion matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
from sklearn.metrics import confusion_matrix
labels = ['f', 't']
cnf_matrix = confusion_matrix(lie_test, Prediction_BL3, labels)
print("The confusion matrix is:")
print(cnf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cnf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(BernModelL3.predict_proba(BVbuilder3_test),2))

#Accuracy score of the model
print ("Score:", BernModelL3.score(BVbuilder3_test, lie_test))

#cross-validation method
scores_BL3 = cross_val_score(BernModelL3, BVectbuilder3, lie, scoring='accuracy', cv=10)
print ("Cross-validated scores:", scores_BL3)

#get the mean of each fold 
print("Accuracy of Model with Cross Validation is:",scores_BL3.mean() * 100)

predictions_cvBL3 = cross_val_predict(BernModelL3, BVectbuilder3, lie,  cv=10)
plt.scatter(lie, predictions_cvBL3)

### Lie Dataset--------------------------------------------------


### Sent Dataset--------------------------------------------------

BernModelSentS3 = BernoulliNB()
BernModelSentS3.fit(BVSbuilder3_train, sent_train)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Prediction_BS3 = BernModelSentS3.predict(BVSbuilder3_test)
print("Bernoulli prediction:\n", Prediction_BS3)
print("Actual:")
print(sent_test)

## confusion matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
from sklearn.metrics import confusion_matrix
labels = ['n', 'p']
cnf_matrix = confusion_matrix(sent_test, Prediction_BS3, labels)
print("The confusion matrix is:")
print(cnf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cnf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(BernModelSentS3.predict_proba(BVSbuilder3_test),2))

#Accuracy score of the model
print ("Score:", BernModelSentS3.score(BVSbuilder3_test, sent_test))

#cross-validation method
scores_BS3 = cross_val_score(BernModelSentS3, BVectS3builder, sent, scoring='accuracy',  cv=10)
print ("Cross-validated scores:", scores_BS3)

#get the mean of each fold 
print("Accuracy of Model with Cross Validation is:",scores_BS3.mean() * 100)

predictions_cvBS3 = cross_val_predict(BernModelSentS3, BVectS3builder, sent, cv=10)
plt.scatter(sent, predictions_cvBS3)

### Sent Dataset--------------------------------------------------