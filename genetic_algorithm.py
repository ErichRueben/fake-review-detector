import pandas as pd
from random import random
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

PARENTS = 5
ROUNDS = 2
MODEL_TRAINING_PERCENTAGE = 0.5


def readData():
    data = pd.read_csv( "test.csv" )
    X_train, X_test, y_train, y_test = train_test_split( 
        data[ 'review' ], 
        data[ 'reality' ], 
        test_size=MODEL_TRAINING_PERCENTAGE, 
        random_state=42 
    )
    return { 'X': X_train, 'y': y_train }, { 'X', X_test, 'y', y_test }


def trainClassifiers( data ):

    logreg = Pipeline( [
        ( 'scaler', StandardScaler() ),
        ( 'logreg', LogisticRegression() )
    ] )
    logreg.fit( data[ 'X' ], data[ 'y' ] )

    kneighbors = Pipeline( [
        ( 'scaler', StandardScaler() ),
        ( 'kneighbors', KNeighborsClassifier() )
    ] )
    kneighbors.fit( data[ 'X' ], data[ 'y' ] )

    svc = Pipeline( [
        ( 'scaler', StandardScaler() ),
        ( 'svc', SVC() )
    ] )
    svc.fit( data[ 'X' ], data[ 'y' ] )

    naivebayes = Pipeline( [
        ( 'scaler', StandardScaler() ),
        ( 'naivebayes', MultinomialNB() )
    ] )
    naivebayes.fit( data[ 'X' ], data[ 'y' ] )

    return { 'logreg' : logreg, 'svc' : svc, 'kneighbors' : kneighbors, 'naivebayes' : naivebayes }


def initializeParents():
    parents = []
    for index in range( PARENTS ):
        parents.append(
            {
                'id' : index,
                'logreg' : random(),
                'kneighbors' : random(),
                'svc' : random(),
                'naivebayes' : random()
            }
        )
    return parents

def main():
    model_training_data, testing_data = readData()
    pipelines = trainClassifiers( model_training_data )


if __name__=='__main__':
    main()