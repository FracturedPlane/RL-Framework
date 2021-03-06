import theano
from theano import tensor as T
import numpy as np
import load_data

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def sgd(cost, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p + (-g * lr)])
    return updates

class NeuralNet(object):
    
    def __init__(self, input, n_in, n_out):

        hidden_size=36
        self._w_h = init_weights((n_in, hidden_size))
        self._b_h = init_weights((hidden_size,))
        self._w_o = init_weights((hidden_size, n_out))
        self._b_o = init_weights((n_out,))
        
        self._w_h_old = init_weights((n_in, hidden_size))
        self._b_h_old = init_weights((hidden_size,))
        self._w_o_old = init_weights((hidden_size, n_out))
        self._b_o_old = init_weights((n_out,))
        # print "Initial W " + str(self._w_o.get_value()) 
        
        self._learning_rate = 0.00004
        self._discount_factor= 0.8
        
        self._weight_update_steps=3000
        self._updates=0
        
        
        State = T.fmatrix()
        ResultState = T.fmatrix()
        Reward = T.fmatrix()
        # Q_val = T.fmatrix()
        
        # model = T.nnet.sigmoid(T.dot(State, self._w) + self._b.reshape((1, -1)))
        # self._model = theano.function(inputs=[State], outputs=model, allow_input_downcast=True)
        py_x = self.model(State)
        y_pred = T.argmax(py_x, axis=1)
        q_val = py_x
        
        # cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
        # delta = ((Reward.reshape((-1, 1)) + (self._discount_factor * T.max(self.model(ResultState), axis=1, keepdims=True)) ) - self.model(State))
        delta = ((Reward + (self._discount_factor * T.max(self.targetModel(ResultState), axis=1, keepdims=True)) ) - T.max(self.model(State), axis=1,  keepdims=True))
        bellman_cost = T.mean( 0.5 * ((delta) ** 2 ))
        # bellman_cost = T.mean( 0.5 * ((delta) ** 2 )) + (T.sum(self._w_h**2) + T.sum(self._b_h ** 2) + 
          #                                              T.sum(self._w_o**2) + T.sum(self._b_o ** 2))

        params = [self._w_h, self._b_h, self._w_o, self._b_o]
        updates = sgd(bellman_cost, params, lr=self._learning_rate)
        
        self._train = theano.function(inputs=[State, Reward, ResultState], outputs=bellman_cost, updates=updates, allow_input_downcast=True)
        self._predict = theano.function(inputs=[State], outputs=y_pred, allow_input_downcast=True)
        self._q_values = theano.function(inputs=[State], outputs=py_x, allow_input_downcast=True)
        self._bellman_error = theano.function(inputs=[State, Reward, ResultState], outputs=delta, allow_input_downcast=True)
        
        
    def model(self, State):
        h = T.nnet.sigmoid(T.dot(State, self._w_h) + self._b_h)
        qyx = T.nnet.sigmoid(T.dot(h, self._w_o) + self._b_o)
        return qyx
    
    def targetModel(self, State):
        h = T.nnet.sigmoid(T.dot(State, self._w_h_old) + self._b_h_old)
        qyx = T.nnet.sigmoid(T.dot(h, self._w_o_old) + self._b_o_old)
        return qyx
    
    def updateTargetModel(self):
        print "Updating target Model"
        self._w_h_old = self._w_h 
        self._b_h_old = self._b_h 
        self._w_o_old = self._w_o 
        self._b_o_old = self._b_o 
    
    def train(self, state, reward, result_state):
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        return self._train(state, reward, result_state)
    
    def predict(self, state):
        return self._predict(state)
    def q_values(self, state):
        return self._q_values(state)
    def bellman_error(self, state, reward, result_state):
        return self._bellman_error(state, reward, result_state)
        
        

if __name__ == "__main__":

    dataset = "controllerTuples.json"
    states, actions, result_states, rewards = load_data.load_data(dataset)
    
    classifier = NeuralNet(states, n_in=9, n_out=9)
    best_error=10000000.0
    
    # print "Initial Model: " + str(classifier._model(states).shape)
    # print "Initial Model: " + str(np.max(classifier._model(states), axis=1, keepdims=True).shape)
    for i in range(20000):
        for start, end in zip(range(0, len(states), 128), range(128, len(states), 128)):
            # cost = train(states[start:end], rewards[start:end], result_states[start:end])
            _states = states[start:end]
            _rewards = rewards[start:end]
            _result_states = result_states[start:end]
            """
            print _states.shape
            print _rewards.shape
            print _result_states.shape
            """
            cost = classifier.train(_states, _rewards, _result_states)
            # print "Cost: " + str(cost)
        if i % 100 == 0: 
            print ""
            error = classifier.bellman_error(states, rewards, result_states)
            error = np.mean(np.fabs(error))
            print "Iteration: " + str(i) + " Cost: " + str(cost) + " Bellman Error: " + str(error)
            print classifier.q_values(states)[:1]
        # print i, np.mean(np.argmax(q_val, axis=1) == predict(states))
        error = classifier.bellman_error(states, rewards, result_states)
        error = np.mean(np.fabs(error))
        # print "Iteration: " + str(i) + " Cost: " + str(cost) + " Bellman Error: " + str(error)
        if error < best_error:
                # print "Saving better model"
                # load_data.save_model(classifier)
                best_error=error
        
    # save the best model
    load_data.save_model(classifier)
    
    model = load_data.load_model()
    
    print (model.predict(states)[:1000])
    print model._w_o.get_value()
    print model.q_values(states)[:50]

