import tensorflow as tf
import numpy as np
from preprocess import get_data
import math

from sklearn.ensemble import RandomForestRegressor

class ModelThree(tf.keras.Model):

    def __init__(self):

        super(ModelThree, self).__init__()

        self.rf = RandomForestRegressor(n_estimators=100)

class ModelOne(tf.keras.Model):

    def __init__(self):

        super(ModelOne, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.batch_size = 100
        self.window_size = 21
        self.rnn_size = 128

        # normal implem
        self.lstm_one = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=True, return_state=True)
        self.dropout_one = tf.keras.layers.Dropout(0.3)
        self.lstm_two = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=False, return_state=True)
        self.dropout_two = tf.keras.layers.Dropout(0.3)
        self.dense_layer_one = tf.keras.layers.Dense(units=32, kernel_initializer="uniform", activation='relu')
        self.dense_layer_two = tf.keras.layers.Dense(units=1, kernel_initializer="uniform", activation='linear')

    @tf.function
    def call(self, inputs):
        output_seq, final_out, final_state = self.lstm_one(inputs)
        output_seq = self.dropout_one(output_seq)
        a, final_out, c = self.lstm_two(output_seq)
        final_out = self.dropout_two(final_out)
        dense_one_out = self.dense_layer_one(final_out)
        return self.dense_layer_two(dense_one_out)

    @tf.function    
    def loss_function(self, logits, labels):
        return tf.math.reduce_mean(tf.keras.losses.MSE(labels, logits))

class ModelTwo(tf.keras.Model):

    def __init__(self):

        super(ModelTwo, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.batch_size = 100
        self.window_size = 9
        self.rnn_size = 128

        # stacked layer 
        self.lstm_one_s = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=True, return_state=True, stateful=True)
        self.dropout_one_s = tf.keras.layers.Dropout(0.3)
        self.lstm_two_s = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=False, return_state=True)
        self.dropout_two_s = tf.keras.layers.Dropout(0.3)
        self.dense_layer_one_s = tf.keras.layers.Dense(units=32, kernel_initializer="uniform", activation='relu')
        self.dense_layer_two_s = tf.keras.layers.Dense(units=1, kernel_initializer="uniform", activation='linear')
    
    def call(self, inputs):
        output_seq, final_out, final_state = self.lstm_one_s(inputs)
        output_seq = self.dropout_one_s(output_seq)
        a, final_out, c = self.lstm_two_s(output_seq)
        final_out = self.dropout_two_s(final_out)
        dense_one_out = self.dense_layer_one_s(final_out)
        return self.dense_layer_two_s(dense_one_out)

    @tf.function    
    def loss_function(self, logits, labels):
        return tf.math.reduce_mean(tf.keras.losses.MSE(labels, logits))

def train(model_one, model_two, train_input, train_labels):
    
    counter = 0
    total_loss_one = 0
    total_loss_two = 0
    num_inputs = np.shape(train_labels)[0]
    num_days = np.shape(train_labels)[1]
    num_batches = num_days - num_days % model_one.batch_size
    for i, stock in enumerate(train_input):
        count = 0
        pointer = 0 
        while pointer < num_batches:
            counter += 1
            start = pointer
            pointer += model_one.batch_size
            batch_inputs = stock[start:pointer, :, :]
            batch_labels = train_labels[i, start:pointer]
            with tf.GradientTape() as tape_one:
                logits_one = model_one.call(batch_inputs)
                loss_one = model_one.loss_function(logits_one, batch_labels) 
                with tf.GradientTape() as tape_two:
                # make new dataset of predictions and real labels 
                    batch_labels = np.reshape(batch_labels,(100, -1))
                    new_data = np.hstack((logits_one, batch_labels)) # shape = (100, 2)
                    predictions = []
                    for j in range(model_two.batch_size - model_two.window_size): 
                        predictions.append(new_data[j:j + model_two.window_size]) 
                    predictions = np.array(predictions) # shape = (91, 9, 2) 
                    labels_two = batch_labels[model_two.window_size:] # shape = (91, 1)
                    logits_two = model_two.call(predictions)
                    loss_two = model_two.loss_function(logits_two, labels_two)
        
            total_loss_one += loss_one
            total_loss_two += loss_two
            gradients_one = tape_one.gradient(loss_one, model_one.trainable_variables)
            model_one.optimizer.apply_gradients(zip(gradients_one, model_one.trainable_variables))
            gradients_two = tape_two.gradient(loss_two, model_two.trainable_variables)
            model_two.optimizer.apply_gradients(zip(gradients_two, model_two.trainable_variables))
        print("I'm working.")

    print('training loss one avg:', total_loss_one/counter)
    print('training loss two avg:', total_loss_two/counter)
    return 
               

def test(model_one, model_two, test_input, test_labels):

    counter = 0
    total_loss_one = 0
    total_loss_two = 0
    num_inputs = np.shape(test_labels)[0]
    num_days = np.shape(test_labels)[1]
    num_batches = num_days - num_days % model_one.batch_size
    for i, stock in enumerate(test_input):
        count = 0
        pointer = 0 
        while pointer < num_batches:
            counter += 1
            start = pointer
            pointer += model_one.batch_size
            batch_inputs = stock[start:pointer, :, :]
            batch_labels = test_labels[i, start:pointer]
            logits_one = model_one.call(batch_inputs)
            loss_one = model_one.loss_function(logits_one, batch_labels) 
            # make new dataset of predictions and real labels 
            batch_labels = np.reshape(batch_labels,(100, -1))
            new_data = np.hstack((logits_one, batch_labels)) # shape = (100, 2)
            predictions = []
            for j in range(model_two.batch_size - model_two.window_size): 
                predictions.append(new_data[j:j + model_two.window_size]) 
            predictions = np.array(predictions) # shape = (91, 9, 2) 
            labels_two = batch_labels[model_two.window_size:] # shape = (91, 1)
            logits_two = model_two.call(predictions)
            loss_two = model_two.loss_function(logits_two, labels_two)
            total_loss_one += loss_one
            total_loss_two += loss_two
    return total_loss_one/counter, total_loss_two/counter

def main():
    
    train_input, train_labels, test_input, test_labels = get_data("./data/prices.csv", "./data/fundamentals.csv")
    model_one = ModelOne()
    model_two = ModelTwo()
    train(model_one, model_two, train_input, train_labels)
    avg_loss_one, avg_loss_two = test(model_one, test_input, test_labels)
    print('testing loss one avg:', avg_loss_one.numpy())
    print('testing loss two avg:', avg_loss_two.numpy())
    
if __name__ == '__main__':
    main()
