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

FILE = 'dataset.csv'
PARENTS = 10
ROUNDS = 100
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
        random_weights = np.random.rand( 4 )
        random_weights /= np.sum( random_weights )
        new_parent = Parent( 
                index, 
                random_weights[ 0 ], 
                random_weights[ 1 ],
                random_weights[ 2 ],
                random_weights[ 3 ]
            )
        parents.append( new_parent )

    return parents


def uniformCrossover( parents, fitness_scores ):
    num_parents, len_chromos = len(parents), len( parents[ 0 ].chromos )
    fitness_prob = np.array( fitness_scores ) / np.sum( fitness_scores )
    offsprings = []

    for index in range( len( parents ) ):
        chromos = []
        for i in range( len_chromos ):
            chosen_parent = np.random.choice( num_parents, p=fitness_prob )
            chromos.append( parents[ chosen_parent ].chromos[ i ] )
        chromos /= sum( chromos )
        offsprings.append( Parent( index, chromos[ 0 ], chromos[ 1 ], chromos[ 2 ], chromos[ 3 ] ) )
    return offsprings


def geneticAlgorithm( pipelines, data ):
    parents = initializeParents()
    fitness_scores = []
    for index in range( ROUNDS ):
        print( f'ROUND { index }:' )
        fitness_scores = []
        for parent in parents:
            fitness_scores.append( parent.evaluateFitness( pipelines, data ) )
        parents = uniformCrossover( parents, fitness_scores )
        print( f'    Average Fitness: { np.mean( fitness_scores ) * 100 }%' )

def main():
    classifier_training_data, genetic_data = readData()
    pipelines = trainClassifiers( classifier_training_data )
    geneticAlgorithm( pipelines,genetic_data )


if __name__=='__main__':
    main()