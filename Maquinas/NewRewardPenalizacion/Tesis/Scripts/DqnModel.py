import warnings
import time
import cv2
import random
import numpy as np
import math as m
import matplotlib.pyplot as plt

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow.keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, concatenate
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
    import tensorflow as tf
    
class DQNModel():
  # --------------------------------------------------------------
  # ----------------------- Initialization -----------------------
  # --------------------------------------------------------------
  def __init__(self, lr=0.01):
    print('Building a new model...')
    # modelos
    self.depthModel = Sequential(name='DepthConvLayers')
    self.outputModel = Sequential(name='OutputLayers')
    # Entradas
    depthInput = Input(shape=(60, 80, 1), name='DepthRaw')
    positionInput = Input(shape=(2,1,), name='positionInfo')
    # Visual model
    self.depthModel.add(Conv2D(8, (8, 8), activation='tanh'))
    self.depthModel.add(BatchNormalization())
    self.depthModel.add(Activation('tanh'))
    self.depthModel.add(MaxPooling2D())
    self.depthModel.add(Conv2D(16, (4, 4), activation='tanh'))
    self.depthModel.add(MaxPooling2D())
    self.depthModel.add(Conv2D(32, (2, 2), activation='tanh'))
    self.depthModel.add(MaxPooling2D())
    self.depthModel.add(Flatten())
    #concatenate
    encodedDepth = self.depthModel(depthInput)
    mergedModel = concatenate([encodedDepth, positionInput],
                                   axis=-1)
    #Salidas
    self.outputModel.add(Dense(100, activation='tanh'))
    self.outputModel.add(Dense(4, activation='linear'))
    output = self.outputModel(mergedModel)

    self.completeModel = Model(name='PepperNavigationModel',
                                inputs=[ depthInput, positionInput],
                                outputs=output)
    self.completeModel.summary()

    self.compile(lr)
    #tf.keras.utils.plot_model(self.completeModel, to_file='DQNModel.png')

  def train(self, state, targetQ):
      #history = self.completeModel.fit(state, targetQ)
      history = self.completeModel.train_on_batch(state, targetQ)
      print('(Training) Loss / Accuracy from batch: ' + str(history))
      return history
  def get_values(self, state):
          qValues = self.completeModel.predict(state)
          return qValues
  def get_action(self, state):
        qValues = self.completeModel.predict(state)
        print(qValues)
        return np.argmax(qValues)
  def save_model(self, name='DQNModel_X'):
        self.completeModel.save(name + '.h5')
  def save_w(self, name='DQNModel_X'):
        # Guardar pesos en el disco
        self.completeModel.save_weights(name + '_weights.h5')
  def load_w(self, name='DQNModel_X'):
        self.completeModel.load_weights(name + '.h5')
  def get_w(self):
        w = self.completeModel.get_weights()
        return w
  def soft_update(self, model, tau=0.01):
        q_model_theta = model.get_w()
        target_model_theta = self.completeModel.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_model_theta, target_model_theta):
            target_weight = target_weight * (1-tau) + q_weight * tau
            target_model_theta[counter] = target_weight
            counter += 1
        self.completeModel.set_weights(target_model_theta)
  def compile(self, lr=0.01):
        # Compile
        adamOpti = Adam(learning_rate=lr)
        self.completeModel.compile(optimizer=adamOpti, loss='mse',
                                   metrics=['accuracy'])