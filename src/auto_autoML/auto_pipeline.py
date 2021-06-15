from auto_autoML.auto_preprocessing.auto_preprocessing import Preprocess
from auto_autoML.auto_preprocessing.auto_preprocessing import get_train_test_data
from tpot import TPOTClassifier, TPOTRegressor
from mlbox.optimisation import Optimiser
from sklearn.model_selection import cross_val_score
from auto_autoML.utils import prettyprint, get_best_automl_package
import pickle


class AutoML:

    def __init__(self, numerical_drift_threshold=0.6, max_cardinality=0.9, prediction_type='classification', scoring='accuracy', n_folds=5, population_size=50):
        self.numerical_drift_threshold = numerical_drift_threshold
        self.max_cardinality = max_cardinality
        self.scoring = scoring
        self.n_folds = n_folds
        self.population_size = population_size
        self.prediction_type = prediction_type


    def optimise(self, datapath, target, split_ratio, stratified, max_evaluations, hyper_param_space):

            self.df = get_train_test_data(datapath, target, split_ratio, stratified=stratified)
            self.preprocess = Preprocess(self.numerical_drift_threshold, self.max_cardinality)
            self.df = self.preprocess.fit_transform(self.df)
            self.results = {'MLBox': {}, 'Tpot': {}}

            #########################################
            #                 MLBox                 #
            #########################################
            self.mlbox_opt = Optimiser(scoring=self.scoring, n_folds=self.n_folds)
            self.ml_box_params = self.mlbox_opt.optimise(hyper_param_space, self.df, max_evaluations)
            self.ml_box_eval = self.mlbox_opt.evaluate(self.ml_box_params, self.df)
            self.results['MLBox']['cross_validation_mean'] = self.ml_box_eval
            self.results['MLBox']['parameters'] = self.ml_box_params


            #########################################
            #                 Tpot                  #
            #########################################
            if self.prediction_type == 'classification':
                self.tpot = TPOTClassifier(generations=self.n_folds, population_size=self.population_size, verbosity=2, random_state=42)

            elif self.prediction_type == 'regression':
                self.tpot = TPOTRegressor(generations=self.n_folds, population_size=self.population_size, verbosity=2, random_state=42)

            self.tpot.fit(self.df['train'], self.df['target'])

            self.results['Tpot']['cross_validation_mean'] = cross_val_score(self.tpot.fitted_pipeline_, self.df['train'], self.df['target'], scoring=self.scoring, cv=5).mean()
            self.results['Tpot']['parameters'] = self.tpot.fitted_pipeline_.get_params()

            self.best_package, self.best_parameters = get_best_automl_package(self.results)

            print('Best results from different auto ML libraries:')
            print(f'{self.best_package}')
            print('----------------------------------------------------------------------------------------------')
            prettyprint(self.results)

            pkl_filename = "hyperparameters.pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(self.best_parameters, file)

            return self.best_parameters







