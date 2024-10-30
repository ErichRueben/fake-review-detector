class Parent:

    def __init__( self, id, logreg, kneighbors, svc, naivebayes ):
        self.id = id
        self.chromos = [ logreg, kneighbors, svc, naivebayes ]
        self.fitness = 0

    def evaluateFitness( self, pipelines, data ):
        X = data[ 'X' ]
        y = data[ 'y' ]

        logreg_pred = pipelines[ 'logreg' ].predict( X )
        kneighbors_pred = pipelines[ 'kneighbors' ].predict( X )
        svc_pred = pipelines[ 'svc' ].predict( X )
        naivebayes_pred = pipelines[ 'naivebayes' ].predict( X )

        fitness = (
            self.chromos[ 0 ] * ( logreg_pred == y ).mean() +
            self.chromos[ 1 ] * ( kneighbors_pred == y ).mean() +
            self.chromos[ 2 ] * ( svc_pred == y ).mean() +
            self.chromos[ 3 ] * ( naivebayes_pred == y ).mean()
        )

        self.fitness = float( fitness )
        # print( f'  { self }' )
        return fitness
    
    def __str__( self ):
        return f"Parent { self.id }\n    Logistic Regression = { self.chromos[ 0 ] }\n    K-Neighbors = { self.chromos[ 1 ] }\n    SVC = { self.chromos[ 2 ] }\n    Naive Bayes = { self.chromos[ 3 ] }\n    Fitness = { self.fitness * 100 }%"