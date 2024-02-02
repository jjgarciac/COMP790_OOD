import os
from catboost import CatBoostClassifier
from catboost import Pool
import numpy as np
import keras

class GBDT(keras.Model):
    def __init__(self,  nensemble=10, 
                        iterations=1000,
                        depth=6, 
                        learning_rate=0.1, 
                        border_count=128, 
                        random_strength=0, 
                        bootstrap_type='Bernoulli',
                        posterior_sampling=True,
                        random_seed=33,
                        subsample=0.5,
                        class_weights=None):
        super().__init__()
        self.ensemble = []
        for e in range(nensemble):
            m = CatBoostClassifier(iterations=iterations,
                        depth=depth,
                        learning_rate=learning_rate,
                        border_count=border_count,
                        random_strength=random_strength,
                        loss_function='Logloss',
                        verbose=False,
                        bootstrap_type='Bernoulli',
                        custom_metric='ZeroOneLoss',
                        posterior_sampling=True,
                        random_seed=random_seed+e,
                        subsample=subsample,
                        class_weights=class_weights)
            self.ensemble.append(m)

    def call(self, x):
        return np.mean(np.stack([m.predict_proba(x.numpy()) for m in self.ensemble]), axis=0)

    def predict(self,x):
        return np.argmax(self(x), axis=-1)
    
    def score(self,x):
        probs = self(x)
        log_probs = -np.log(probs + 1e-10)
        return np.sum(probs * log_probs, axis=1)

    def fit(self, X, y, validation_data=(), use_best_model=True, cat_features=None):
        train_pool = Pool(X, y, cat_features=cat_features)
        val_pool = Pool(validation_data[0], validation_data[1], cat_features=cat_features)        
        for m in self.ensemble:
            m.fit(train_pool, eval_set=val_pool, use_best_model=use_best_model) # use_best_model=False Required by sglb
            print("best iter ", m.get_best_iteration())
            print("best score ", m.get_best_score())

    def save(self, path):
        for i, m in enumerate(self.ensemble):
            m.save_model(os.path.join(path, f"saved_model/model_{i}"))

def grid_search(X, y, random_seed):
    """Searches for the optimal hyperparameters.
    Args:
        X: Data matrix (NxD array)
        y: Targets (N array)
        random_seed: (Integer)
        return_any: Returns a set of hyperparameters. 

    Returns:
        Dictionary with optimal hyperparameters found.
    """

    neg, pos = np.bincount(y)
    total = neg+pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weights = {0: weight_for_0, 1: weight_for_1}    
    
    grid = {'learning_rate': [0.01, 0.1, 1.0],
            'depth': [1, 3, 6, 10],
            'subsample': [0.25, 0.5, 0.75],
            'iterations':[1000],
            'border_count':[128],
            'random_strength':[0],
            'bootstrap_type':['Bernoulli'],
            'posterior_sampling':[True],
            'random_seed':[random_seed],
            'class_weights': [[class_weights[0], class_weights[1]]],
        }
    
    model = CatBoostClassifier(loss_function='Logloss', verbose=False, eval_metric='Accuracy')
    hyperparams = model.grid_search(grid, X, y, refit=False, search_by_train_test_split=True)['params']
    hyperparams['nensemble']=10

    return hyperparams
