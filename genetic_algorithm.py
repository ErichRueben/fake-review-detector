import pandas as pd
import numpy as np
from random import random
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
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
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform( data[ FEATURE_COLUMN ] )
    print( X.shape )
    X_train, X_genetic, y_train, y_genetic = train_test_split( 
        X, 
        data[ TARGET_COLUMN ], 
        test_size=MODEL_TRAINING_PERCENTAGE, 
        random_state=42 
    )
    return { 'X': X_train, 'y': y_train }, { 'X': X_genetic, 'y': y_genetic }


def trainClassifiers( data ):

    logreg = Pipeline( [
        ( 'scaler', MaxAbsScaler() ),
        ( 'logreg', LogisticRegression() )
    ] )
    logreg.fit( data[ 'X' ], data[ 'y' ] )

    kneighbors = Pipeline( [
        ( 'scaler', MaxAbsScaler() ),
        ( 'kneighbors', KNeighborsClassifier( n_neighbors=2 ) )
    ] )
    kneighbors.fit( data[ 'X' ], data[ 'y' ] )

    svc = Pipeline( [
        ( 'scaler', MaxAbsScaler() ),
        ( 'svc', SVC() )
    ] )
    svc.fit( data[ 'X' ], data[ 'y' ] )

    naivebayes = Pipeline( [
        ( 'scaler', MaxAbsScaler() ),
        ( 'naivebayes', MultinomialNB() )
    ] )
    naivebayes.fit( data[ 'X' ], data[ 'y' ] )
    
    return { 'logreg' : logreg, 'svc' : svc, 'kneighbors' : kneighbors, 'naivebayes' : naivebayes }


def initializeParents():
    parents = []
    for index in range( PARENTS ):
        random_weights = np.random.dirichlet( np.ones( 4 ) )
        parents.append(
            {
                'id' : index,
                'logreg' : random_weights[ 0 ],
                'kneighbors' : random_weights[ 1 ],
                'svc' : random_weights[ 2 ],
                'naivebayes' : random_weights[ 3 ]
            }
        )
    return parents


def evaluateFitness( parent, pipelines, data ):
    X = data[ 'X' ]
    y = data[ 'y' ]

    logreg_pred = pipelines[ 'logreg' ].predict( X )
    kneighbors_pred = pipelines[ 'kneighbors' ].predict( X )
    svc_pred = pipelines[ 'svc' ].predict( X )
    naivebayes_pred = pipelines[ 'naivebayes' ].predict( X )

    fitness = (
        parent[ 'logreg' ] * ( logreg_pred == y ).mean() +
        parent[ 'kneighbors' ] * ( kneighbors_pred == y ).mean() +
        parent[ 'svc' ] * ( svc_pred == y ).mean() +
        parent[ 'naivebayes' ] * ( naivebayes_pred == y ).mean()
    )

    print( f'Parent { parent[ 'id' ] } has fitness { fitness }' )
    return { 'id' : parent[ 'id' ], 'fitness' : fitness }


def getBestParents( fitness ):
    best_parents = []
    for _ in range( int( PARENTS / 2 ) ):
        sorted_fitness = sorted( fitness, key=lambda x : x[ 'fitness' ], reverse=True )
        best_parents = sorted_fitness[ :int( PARENTS / 2 ) ]
    return best_parents


def geneticAlgorithm( pipelines, data ):
    parents = initializeParents()

    for index in range( ROUNDS ):
        fitness_result = []
        for parent in parents:
            fitness_result.append( evaluateFitness( parent, pipelines, data ) )
        best_parents = getBestParents( fitness_result )
        print( 'Best Parents:' )
        print( best_parents )


def main():
    classifier_training_data, genetic_data = readData()
    pipelines = trainClassifiers( classifier_training_data )
    geneticAlgorithm( pipelines,genetic_data )


if __name__=='__main__':
    main()