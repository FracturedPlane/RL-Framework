import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from util.nn import weight_norm



# elu
def elu_mine(x):
    return theano.tensor.switch(x > 0, x, theano.tensor.expm1(x))

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface

class ForwardDynamicsDenseNetworkDropoutTesting(ModelInterface):
    
    def __init__(self, state_length, action_length, state_bounds, action_bounds, settings_):

        super(ForwardDynamicsDenseNetworkDropoutTesting,self).__init__(state_length, action_length, state_bounds, action_bounds, 0, settings_)
        
        batch_size=32
        # data types for model
        self._State = T.matrix("State")
        self._State.tag.test_value = np.random.rand(batch_size,self._state_length)
        self._ResultState = T.matrix("ResultState")
        self._ResultState.tag.test_value = np.random.rand(batch_size,self._state_length)
        self._Reward = T.col("Reward")
        self._Reward.tag.test_value = np.random.rand(self._batch_size,1)
        self._Action = T.matrix("Action")
        self._Action.tag.test_value = np.random.rand(batch_size, self._action_length)
        # create a small convolutional neural network
        input = lasagne.layers.InputLayer((None, self._state_length), self._State)
        self._stateInputVar = input.input_var
        actionInput = lasagne.layers.InputLayer((None, self._action_length), self._Action)
        self._actionInputVar = actionInput.input_var
        
        insert_action_later = True
        double_insert_action = False
        add_layers_after_action = False
        if (not insert_action_later or (double_insert_action)):
            input = lasagne.layers.ConcatLayer([input, actionInput])
        ## Activation types
        # activation_type = elu_mine
        # activation_type=lasagne.nonlinearities.tanh
        activation_type=lasagne.nonlinearities.leaky_rectify
        # activation_type=lasagne.nonlinearities.rectify
        # network = lasagne.layers.DropoutLayer(input, p=self._dropout_p, rescale=True)
        """
        network = lasagne.layers.DenseLayer(
                input, num_units=128,
                nonlinearity=activation_type)
        network = weight_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        layersAct = [network]
        
        network = lasagne.layers.DenseLayer(
                network, num_units=64,
                nonlinearity=activation_type)
        network = weight_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        layersAct.append(network)
        network = lasagne.layers.ConcatLayer([layersAct[1], layersAct[0]])
        
        network = lasagne.layers.DenseLayer(
                network, num_units=32,
                nonlinearity=activation_type)
        network = weight_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        layersAct.append(network)
        network = lasagne.layers.ConcatLayer([layersAct[2], layersAct[1], layersAct[0]])
        ## This can be used to model the reward function
        self._reward_net = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
                # print ("Initial W " + str(self._w_o.get_value()) )
        """
        
        network = lasagne.layers.DenseLayer(
                input, num_units=128,
                nonlinearity=activation_type)
        # network = weight_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        # layersAct = [network]
        
        if ( insert_action_later ):
            ### Lets try adding the action input later on in the network
            if ( add_layers_after_action ):
                networkA = lasagne.layers.DenseLayer(
                        actionInput, num_units=32,
                        nonlinearity=activation_type)
                network = lasagne.layers.ConcatLayer([network, networkA])
            else:
                network = lasagne.layers.ConcatLayer([network, actionInput])
        
        network = lasagne.layers.DenseLayer(
                network, num_units=64,
                nonlinearity=activation_type)
        # network = weight_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        
        # layersAct.append(network)
        # network = lasagne.layers.ConcatLayer([layersAct[1], layersAct[0]])
        
        network = lasagne.layers.DenseLayer(
                network, num_units=32,
                nonlinearity=activation_type)
        # network = weight_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        
        
        # layersAct.append(network)
        # network = lasagne.layers.ConcatLayer([layersAct[2], layersAct[1], layersAct[0]])
        # network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        network = lasagne.layers.DenseLayer(
                network, num_units=8,
                nonlinearity=activation_type)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        """
        network = lasagne.layers.DenseLayer(
                network, num_units=8,
                nonlinearity=activation_type)
        """
        ## This can be used to model the reward function
        self._reward_net = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
                # print ("Initial W " + str(self._w_o.get_value()) )
                  
        # networkAct = lasagne.layers.DropoutLayer(input, p=self._dropout_p, rescale=True)
        networkAct = lasagne.layers.DenseLayer(
                input, num_units=256,
                nonlinearity=activation_type)
        networkAct = weight_norm(networkAct)
        networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        layersAct = [networkAct]
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=128,
                nonlinearity=activation_type)
        networkAct = weight_norm(networkAct)
        networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        
        if ( insert_action_later ):
            ### Lets try adding the action input later on in the network
            if ( add_layers_after_action ):
                networkActA = lasagne.layers.DenseLayer(
                    actionInput, num_units=64,
                    nonlinearity=activation_type)
                networkAct = lasagne.layers.ConcatLayer([networkAct, networkActA])
            else:
                networkAct = lasagne.layers.ConcatLayer([networkAct, actionInput])
            
        
        layersAct.append(networkAct)
        networkAct = lasagne.layers.ConcatLayer([layersAct[1], layersAct[0]])
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=128,
                nonlinearity=activation_type)
        networkAct = weight_norm(networkAct)
        networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        layersAct.append(networkAct)
        networkAct = lasagne.layers.ConcatLayer([layersAct[2], layersAct[1], layersAct[0]])
    
        self._forward_dynamics_net = lasagne.layers.DenseLayer(
                networkAct, num_units=self._state_length,
                nonlinearity=lasagne.nonlinearities.linear)
                # print ("Initial W " + str(self._w_o.get_value()) )
                
        if (('use_stochastic_forward_dynamics' in self._settings) and 
            self._settings['use_stochastic_forward_dynamics']):
            with_std = lasagne.layers.DenseLayer(
                    networkAct, num_units=self._state_length,
                    nonlinearity=theano.tensor.nnet.softplus)
            self._forward_dynamics_net = lasagne.layers.ConcatLayer([self._forward_dynamics_net, with_std], axis=1)
                
        self._states_shared = theano.shared(
            np.zeros((batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._actions_shared = theano.shared(
            np.zeros((batch_size, self._action_length), dtype=theano.config.floatX),
            )
        
        self._rewards_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))
        
