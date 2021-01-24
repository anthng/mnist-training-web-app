import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, entropy_sampling

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import time
from pathlib import Path

ROOT_DIR = str(Path(__file__).parent.parent)



class DL:
    @staticmethod
    def LeNet(lr):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(8,8,1)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2),padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
        model.add(LeakyReLU(alpha=0.1))                  
        model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128, activation='linear'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Dropout(0.4))
        model.add(Dense(10, activation='softmax'))
        
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr),metrics=['accuracy'])

        return model


class BaseModel:
    def __init__(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=128, lr=1e-3, estimator=None):
        
        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.estimator = estimator
 
    def train(self):
        pass

    def predict(self, X):
        pass

    def evaluate(self, X_test, y_test):
        pass

class PassiveLearner(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=128, lr=1e-3):
        super().__init__(X_train, y_train, X_test, y_test, epochs, batch_size, lr)

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        BaseModel.estimator = DL.LeNet(self.lr)

    def train(self):
        model_train = super().estimator.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), batch_size=self.batch_size, epochs=self.epochs, verbose=1)
        
    def predict(self, X):
        y_prob = super().estimator.predict(X)
        y_classes = y_prob.argmax(axis=-1)

        return y_classes

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score
        
        y_classes = self.predict(X_test)
        return accuracy_score(y_test.argmax(axis=-1), y_classes)

class CustomAcitveLearner(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=128, lr=1e-3, n_initial=100, n_queries=100, query_strategy=uncertainty_sampling, estimator=None):
        super().__init__(X_train, y_train, X_test, y_test, epochs, batch_size, lr)

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.n_initial = n_initial
        self.n_queries = n_queries
        self.query_strategy = query_strategy

        initial_idx = np.random.choice(range(len(self.X_train)), size=self.n_initial, replace=False)
        self.__X_initial = self.X_train[initial_idx]
        self.__y_initial = self.y_train[initial_idx]
        
        self.__X_pool = np.delete(self.X_train, initial_idx, axis=0)
        self.__y_pool = np.delete(self.y_train, initial_idx, axis=0)

        self.learner = ActiveLearner(
            estimator=DL.LeNet(self.lr),
            query_strategy=self.query_strategy,
            X_training=self.__X_initial, y_training=self.__y_initial,
            verbose=1
        )

        BaseModel.estimator = self.learner        

    def train(self):
        performances = [self.evaluate(self.X_test, self.y_test)]
        for idx in range(self.n_queries):
            try:
                query_idx, query_instance = self.learner.query(self.__X_pool, verbose=0)
            except:
                break

            placeholder = st.empty()
            with plt.style.context('seaborn-white'):
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title('Digit to label')
                plt.imshow(query_instance.reshape(8, 8))
                plt.subplot(1, 2, 2)
                plt.title('Accuracy of your model')
                plt.plot(range(idx+1), performances)
                plt.scatter(range(idx+1), performances)
                plt.xlabel('number of queries')
                plt.ylabel('accuracy')
                
                plt.savefig('../buf.png', format='png')

                with placeholder.beta_container():
                    st.image('../buf.png', use_column_width=True)
                    time.sleep(0.5)
                plt.close()
            placeholder.empty()
            
            self.learner.teach(
                X=self.__X_pool[query_idx], y=self.__y_pool[query_idx], epochs=self.epochs, batch_size=self.batch_size, verbose=0
            )
            self.__X_pool = np.delete(self.__X_pool, query_idx, axis=0)
            self.__y_pool = np.delete(self.__y_pool, query_idx, axis=0)

            model_accuracy = self.evaluate(self.X_test, self.y_test)
            performances.append(model_accuracy)
                        
            # with st.beta_container():
            #     info = 'Accuracy after query {n}: {acc:0.4f}'.format(n=idx + 1, acc=model_accuracy)
            #     st.write(info)

        return performances

    def predict(self, X):
        y_prob = super().estimator.predict(X)
        y_classes = y_prob.argmax(axis=-1)

        return y_classes

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score
        
        y_classes = self.predict(X_test)
        return accuracy_score(y_test.argmax(axis=-1), y_classes)
