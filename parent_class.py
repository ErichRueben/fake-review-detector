class Parent:

    def __init__( self, id, logreg, kneighbors, svc, naivebayes ):
        self.id = id
        self.logreg = logreg
        self.kneighbors = kneighbors
        self.svc = svc
        self.naivebayes = naivebayes

    def evaluateFitness( self, pipelines, data ):
        X = data[ 'X' ]
        y = data[ 'y' ]

        logreg_pred = pipelines[ 'logreg' ].predict( X )
        kneighbors_pred = pipelines[ 'kneighbors' ].predict( X )
        svc_pred = pipelines[ 'svc' ].predict( X )
        naivebayes_pred = pipelines[ 'naivebayes' ].predict( X )

        fitness = (
            self.logreg * ( logreg_pred == y ).mean() +
            self.kneighbors * ( kneighbors_pred == y ).mean() +
            self.svc * ( svc_pred == y ).mean() +
            self.naivebayes * ( naivebayes_pred == y ).mean()
        )

        print( f'Parent { self.id } has fitness { fitness }' )

        return fitness