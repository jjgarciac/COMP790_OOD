import tensorflow as tf
import os

tfk = tf.keras
tfkl = tf.keras.layers
tfkr = tf.keras.regularizers

class DeepEnsemble(tfk.Model):
    def __init__(self, nensemble, nout, nhidden, learning_rate, dropout=.0, reg=.01):
        super(DeepEnsemble, self).__init__()
        self.learning_rate=learning_rate
        self.ensemble = []
        for m in range(nensemble):
            layers = [tfkl.Dropout(dropout), 
                    tfkl.Dense(nhidden[0], activation='relu', kernel_regularizer=tfkr.l2(reg))]
            for hsize in nhidden[1:]:
                layers += [tfkl.Dropout(dropout), 
                        tfkl.Dense(hsize, activation='relu', kernel_regularizer=tfkr.l2(reg))]
            layers += [tfkl.Dropout(dropout), 
                    tfkl.Dense(nout, activation='softmax', kernel_regularizer=tfkr.l2(reg))]
            self.ensemble += [tfk.Sequential(layers)]

    def call(self, xs):
        return tf.reduce_mean(tf.stack([m(xs) for m in self.ensemble]), axis=0)
    
    def H(self, p):
        p = tf.where(tf.abs(p) < 1e-7, 0. * tf.ones_like(p), p)
        return tf.reduce_sum(-tf.math.log(p+1e-9)*p, axis=-1, keepdims=True)

    def data_uncertainty(self, xs):
        # UPDATE: THIS IS OLD and WRONG
        return tf.reduce_mean(tf.concat([self.H(tf.reduce_mean(m(xs), axis=1)) for m in self.ensemble], axis=-1), axis=-1)

    def total_uncertainty(self, x):
        return self.H(self(x))

    def model_uncertainty(self, x):
        return self.total_uncertainty(x)[:,0] - self.data_uncertainty(x)
    
    def compile(self, *args, **kwargs):
        for m in self.ensemble:
            m.compile(loss='sparse_categorical_crossentropy', 
                       optimizer=tfk.optimizers.RMSprop(self.learning_rate), 
                       metrics=['accuracy'])

    def fit(self, *args, **kwargs):
        for i, m in enumerate(self.ensemble):
            print(f"Fitting model {i} \n")
            m.fit(*args, **kwargs)
            
    def predict(self, x):
        return tf.math.argmax(self(x), axis=-1)

    def save(self, path):
        for i, m in enumerate(self.ensemble):
            m.save(os.path.join(path, f"model_{i}"))