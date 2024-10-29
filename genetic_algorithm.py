import pandas as pd
import numpy as np
from parent_class import Parent
from random import random
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

FILE = 'fake reviews dataset.csv'
PARENTS = 10
ROUNDS = 1
MODEL_TRAINING_PERCENTAGE = 0.5
TARGET_COLUMN = 'label'
FEATURE_COLUMN = 'text_'


def readData():
    data = pd.read_csv( FILE )[ :1000 ]
    X_train, X_genetic, y_train, y_genetic = train_test_split( 
        data[ FEATURE_COLUMN ], 
        data[ TARGET_COLUMN ], 
        test_size=MODEL_TRAINING_PERCENTAGE, 
        random_state=42 
    )
    return { 'X': X_train, 'y': y_train }, { 'X': X_genetic, 'y': y_genetic }


def trainClassifiers( data ):
    logreg = Pipeline( [
        ( 'tfidf', TfidfVectorizer() ),
        ( 'logreg', LogisticRegression() )
    ] )
    logreg.fit( data[ 'X' ], data[ 'y' ] )

    kneighbors = Pipeline( [
        ( 'tfidf', TfidfVectorizer() ),
        ( 'kneighbors', KNeighborsClassifier() )
    ] )
    kneighbors.fit( data[ 'X' ], data[ 'y' ] )

    svc = Pipeline( [
        ( 'tfidf', TfidfVectorizer() ),
        ( 'svc', SVC() )
    ] )
    svc.fit( data[ 'X' ], data[ 'y' ] )

    naivebayes = Pipeline( [
        ( 'tfidf', TfidfVectorizer() ),
        ( 'naivebayes', MultinomialNB() )
    ] )
    naivebayes.fit( data[ 'X' ], data[ 'y' ] )
    
    return { 'logreg' : logreg, 'svc' : svc, 'kneighbors' : kneighbors, 'naivebayes' : naivebayes }


def initializeParents():
    parents = []
    for index in range( PARENTS ):
        random_weights = np.random.dirichlet( np.ones( 4 ) )
        parents.append( 
            Parent( 
                index, 
                random_weights[ 0 ], 
                random_weights[ 1 ],
                random_weights[ 2 ],
                random_weights[ 3 ]
            ) 
        )

    return parents


def uniformCrossover():
    # To Do
    return


def geneticAlgorithm( pipelines, data ):
    parents = initializeParents()

    for index in range( ROUNDS ):
        fitness_scores = []
        for parent in parents:
            fitness_scores.append( parent.evaluateFitness( pipelines, data ) )
        parents = uniformCrossover()


def main():
    classifier_training_data, genetic_data = readData()
    pipelines = trainClassifiers( classifier_training_data )
    geneticAlgorithm( pipelines,genetic_data )


if __name__=='__main__':
    main()