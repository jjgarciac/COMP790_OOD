import tensorflow as tf
import optuna

tfk = tf.keras
tfkl = tf.keras.layers
tfkr = tf.keras.regularizers

class MCDropout(tfkl.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

class MCDnet(tfk.Model):
    def __init__(self, nout, 
                 nhidden, 
                 N, 
                 dropout=.05, 
                 tau=1.0, 
                 lengthscale=1e-2, 
                 epochs=10, 
                 batch_size=128, 
                 learning_rate=.01):        
        super(MCDnet, self).__init__()
        
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.epochs=epochs
        
        reg = lengthscale**2 * (1 - dropout) / (2. * N * tau)
        layers = [MCDropout(dropout), 
                  tfkl.Dense(nhidden[0], activation='relu', kernel_regularizer=tfkr.l2(reg))]
        for hsize in nhidden[1:]:
            layers += [MCDropout(dropout), 
                       tfkl.Dense(hsize, activation='relu', kernel_regularizer=tfkr.l2(reg))]
        layers += [MCDropout(dropout), 
                  tfkl.Dense(nout, activation='softmax', kernel_regularizer=tfkr.l2(reg))]
        self.model = tfk.Sequential(layers)
    
    def call(self, x):
        return self.model(x, training=True)
    
    def H(self, p):
        p = tf.where(tf.abs(p) < 1e-7, 0. * tf.ones_like(p), p)
        return tf.reduce_sum(-tf.math.log(p+1e-9)*p, axis=-1)
    
    def score(self, x, s=10):
        tmp = tf.stack([self.model(x, training=True) for i in range(s)])
        return self.H(tf.reduce_mean(tmp, axis=0))

    def data_uncertainty(self, x, s=10):
        return tf.reduce_mean(tf.concat([self.H(tf.reduce_mean(self(x), axis=1)) for i in range(s)], axis=-1), axis=-1)

    def total_uncertainty(self, x, s=10):
        tmp = tf.stack([self.model(x, training=True) for i in range(s)])
        return self.H(tf.reduce_mean(tmp, axis=0))

    def model_uncertainty(self, x, s=10):
        return self.total_uncertainty(x)[:,0] - self.data_uncertainty(x)
    
    def predict(self, x, s=10):
        tmp = tf.stack([self.model(x, training=True) for i in range(s)])
        return tf.math.argmax(tf.reduce_mean(tmp, axis=0), axis=-1)
    
    def fit(self, X, y, validation_data, *args, **kwargs):
        neg, pos = np.bincount(y)
        total = neg+pos
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weights = {0: weight_for_0, 1: weight_for_1}
        self.model.fit(X, y, validation_data, epochs=self.epochs, batch_size=self.batch_size, 
                       class_weights=class_weights, *args, **kwargs)
        
    def compile(self, *args, **kwargs):
        self.model.compile(loss='sparse_categorical_crossentropy', 
                           optimizer=tfk.optimizers.RMSprop(learning_rate=self.learning_rate), 
                           metrics=["accuracy"], *args, **kwargs)
        
def grid_search(X, y, X_val=None, y_val=None, random_seed=11):
    """Searches for the optimal hyperparameters.
    Args:
        X: Train data matrix (NxD array)
        y: Train targets (N array)
        X_val: Validation data matrix (NxD array)
        y_val: Validation targets (N array)
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
    
    nout = np.max(y)+1
    
    def objective(trial):
        epochs=trial.suggest_categorical("epochs", [1,2,3])
        batch_size=trial.suggest_categorical("batch_size", [1,2,3])
        ndepth=trial.suggest_categorical("ndepth", [1,2,3])
        nunits=trial.suggest_categorical("nunits", [100, 500, 1000])
        nhidden=np.ones(ndepth)*nunits
        dropout=trial.suggest_categorical("dropout", [0.005, .05, .01, .1])
        tau=trial.suggest_categorical("tau", [0.02, 0.2, 2.0, 20.0, 200.0])
        lengthscale=trial.suggest_categorical("lengthscale", [1e-2])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

        model = MCDnet(nout, nhidden, len(X), dropout, tau, lengthscale=lengthscale)
        
        model.compile(loss='sparse_categorical_crossentropy', 
                            optimizer=tfk.optimizers.RMSprop(learning_rate=learning_rate), metrics=["accuracy"])
        model.fit(X, y, validation_data=(X_val, y_val), class_weight=class_weights, epochs=epochs, batch_size=batch_size)

        # Evaluate the model accuracy on the validation set.
        score = model.evaluate(X_val, y_val, verbose=0)
        return score[1]
    
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    hyperparams = study.best_trial.params   

    return hyperparams