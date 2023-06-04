import math
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import keras.backend as K
import keras
import data as d
import os
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier


class AneurysmSequence(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x_set, self.y_set = x_set, y_set
        indices = self.negative_sample()
        self.x, self.y = x_set[indices], y_set[indices]
        self.batch_size = batch_size
        self.epoch = 1

    def negative_sample(self):
        healthy_indices = np.where(self.y_set == 0)[0]
        aneurysm_indices = np.where(self.y_set == 1)[0]
        aneurysm_count = len(aneurysm_indices)
        selected_healthy_indices = np.random.choice(healthy_indices, min(aneurysm_count, len(healthy_indices)),
                                                    replace=False)
        indices = np.concatenate((aneurysm_indices, selected_healthy_indices))
        np.random.shuffle(indices)

        return indices

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.epoch % 7 == 0:
            indices = self.negative_sample()
            self.x, self.y = self.x_set[indices], self.y_set[indices]

        self.epoch += 1


def createModel():
    model = Sequential()
    model.add(Flatten(input_shape=(60, 60)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    # TODO Rano zaustavljanje, hyperopt
    X, y = d.loadData()
    X = X[:, 2, :, :]
    y = y[:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, shuffle=True)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train,
                                                                    shuffle=True)
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_validation = scaler.transform(X_validation.reshape(-1, X_validation.shape[-1])).reshape(X_validation.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # params = {
    #     'epochs': [10, 20, 30],
    #     'batch_size': [8, 16, 32, 64]
    # }
    #
    # grid = GridSearchCV(model, params, scoring='f1')
    # grid_result = grid.fit(X_train, y_train, verbose=2)
    #
    # print(grid.best_score_)
    # print(grid.best_params_)

    model = createModel()
    # model = KerasClassifier(createModel)
    model.fit(AneurysmSequence(X_train, y_train, batch_size=16), epochs=30)

    validation_loss, validation_acc = model.evaluate(X_validation, y_validation)
    print('Validation Loss:', validation_loss)
    print('Validation Accuracy:', validation_acc)

    if not os.path.exists("models"):
        os.makedirs("models")

    model.save(os.path.join("models", "model.h5"))
    model.summary()

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    print(f"f1 score: {f1_score(y_test, y_pred)}")
    print(cm)
