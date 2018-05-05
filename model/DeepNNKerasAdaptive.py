
import theano
from theano import tensor as T
import numpy as np
# import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten, Lambda
from keras.layers.convolutional import Conv1D
from keras.layers.merge import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface


def getKerasActivation(type_name):
    """
        Compute a particular type of actiation to use
    """
    import keras.layers
    
    if (type_name == 'leaky_rectify'):
        return keras.layers.LeakyReLU(alpha=0.01)
    if (type_name == 'relu'):
        return Activation('relu')
    if (type_name == 'tanh'):
        return Activation('tanh')
    if (type_name == 'linear'):
        return Activation('linear')
    if (type_name == 'elu'):
        return keras.layers.ELU(alpha=1.0)
    if (type_name == 'softplus'):
        return Activation('softplus')
    else:
        print ("Activation type unknown: ", type_name)
        sys.exit()

class DeepNNKerasAdaptive(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepNNKerasAdaptive,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        ### Apparently after the first layer the patch axis is left out for most of the Keras stuff...
        input = Input(shape=(self._state_length,))
        # input.trainable = True
        print ("Input ",  input)
        
        layer_sizes = self._settings['critic_network_layer_sizes']
        
        print ("Network layer sizes: ", layer_sizes)
        for i in range(len(layer_sizes)):
            # network = Dense(layer_sizes[i], init='uniform')(input)
            network = Dense(layer_sizes[i],
                            kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(input)
            network = getKerasActivation(self._settings['activation_type'])(network)
            
        network= Dense(1,
                       kernel_regularizer=regularizers.l2(self._settings['critic_regularization_weight']))(network)
        network = Activation('linear')(network)
            
        self._critic = Model(input=input, output=network)

        layer_sizes = self._settings['policy_network_layer_sizes']
        
        inputAct = Input(shape=(self._state_length, ))
        print ("Input ",  inputAct)
        print ("Network layer sizes: ", layer_sizes)
        for i in range(len(layer_sizes)):
            # networkAct = Dense(layer_sizes[i], init='uniform')(inputAct)
            networkAct = Dense(layer_sizes[i], 
                               kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(inputAct)
            networkAct = getKerasActivation(self._settings['activation_type'])(networkAct)
        # inputAct.trainable = True
        print ("Network: ", networkAct)         
        
        networkAct = Dense(self._action_length, kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
        networkAct = getKerasActivation(self._settings['last_policy_layer_activation_type'])(networkAct)
        self._actor = Model(input=inputAct, output=networkAct)
        

    ### Setting network input values ###    
    def setStates(self, states):
        pass
        # self._states_shared.set_value(states)
    def setActions(self, actions):
        pass
        # self._actions_shared.set_value(actions)
    def setResultStates(self, resultStates):
        # self._next_states_shared.set_value(resultStates)
        pass
    def setRewards(self, rewards):
        # self._rewards_shared.set_value(rewards)
        pass
    def setTargets(self, targets):
        pass
        # self._targets_shared.set_value(targets)ModelInterface):
    