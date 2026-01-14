import sys
sys.path.append('..')
from utils import *

import argparse
from tensorflow import keras

"""
NeuralNet for the game of LKID
"""
class LKIDNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net - input is flattened 5x5 board
        self.input_boards = keras.Input(shape=(self.board_x * self.board_y,))    # s: batch_size x 25

        # Reshape to 5x5x1 for convolution
        x_image = keras.layers.Reshape((self.board_x, self.board_y, 1))(self.input_boards)  # batch_size x 5 x 5 x 1
        
        # Convolutional layers
        h_conv1 = keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=3)(keras.layers.Conv2D(args.num_channels, 3, padding='same')(x_image)))
        h_conv2 = keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=3)(keras.layers.Conv2D(args.num_channels, 3, padding='same')(h_conv1)))
        h_conv3 = keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=3)(keras.layers.Conv2D(args.num_channels, 3, padding='same')(h_conv2)))
        h_conv4 = keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=3)(keras.layers.Conv2D(args.num_channels, 3, padding='valid')(h_conv3)))
        
        # Flatten and dense layers
        h_conv4_flat = keras.layers.Flatten()(h_conv4)       
        s_fc1 = keras.layers.Dropout(args.dropout)(keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=1)(keras.layers.Dense(512)(h_conv4_flat))))
        s_fc2 = keras.layers.Dropout(args.dropout)(keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=1)(keras.layers.Dense(256)(s_fc1))))
        
        # Output layers
        self.pi = keras.layers.Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # policy: batch_size x action_size
        self.v = keras.layers.Dense(1, activation='tanh', name='v')(s_fc2)                        # value: batch_size x 1

        self.model = keras.Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=keras.optimizers.Adam(learning_rate=args.lr))
