# -*- coding: utf-8 -*-

vecclassesnamespt=[
                    'carga',
                    'comprovacao de disponibilidade',
                    'controle de geracao',
                    'controle de tensao',
                    'controle de transmissao',
                    'conversora',
                    'falha de supervisao',
                    'hidrologia',
                    'horario',
                    'sem informacao',
                    'sgi',
                    'teste de comunicacao'
                ]

vecclassesnamesen=[
                    'load',
                    'proof of availability',
                    'generation control',
                    'voltage control',
                    'transmission control',
                    'converter',
                    'supervision failure',
                    'hydrology',
                    'time confirmation',
                    'no data',
                    'interventions',
                    'communication test'
                ]
dicclassestonum={
                'carga':0,
                'comprovacao de disponibilidade':1,
                'controle de geracao':2,
                'controle de tensao':3,
                'controle de transmissao':4,
                'conversora':5,
                'falha de supervisao':6,
                'hidrologia':7,
                'horario':8,
                'sem informacao':9,
                'sgi':10,
                'teste de comunicacao':11
            }
dicnumtoclasses={
                0:'carga',
                1:'comprovacao de disponibilidade',
                2:'controle de geracao',
                3:'controle de tensao',
                4:'controle de transmissao',
                5:'conversora',
                6:'falha de supervisao',
                7:'hidrologia',
                8:'horario',
                9:'sem informacao',
                10:'sgi',
                11:'teste de comunicacao'
            }
vecmodelsnames=['ComplementNB',
                'LinearSVC', 
                'SGDClassifier', 
                'KNeighborsClassifier', 
                'MLPClassifier', 
                'RandomForestClassifier']