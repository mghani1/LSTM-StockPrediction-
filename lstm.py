import tensorflow as tf
import numpy as np
from preprocess import get_data
import math

from sklearn.ensemble import RandomForestRegressor

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


def train(model_one, train_input, train_labels):
    
    counter = 0
    num_inputs = np.shape(train_labels)[0]
    num_days = np.shape(train_labels)[1]
    num_batches = num_days - num_days % model_one.batch_size
    total_loss = 0
    for i, stock in enumerate(train_input):
        count = 0
        loss_one_sum = 0
        pointer = 0 
        while pointer < num_batches:
            count += 1
            counter += 1
            start = pointer
            pointer += model_one.batch_size
            batch_inputs = stock[start:pointer, :, :]
            batch_labels = train_labels[i, start:pointer]
            with tf.GradientTape() as tape_one:
                logits_one = model_one.call(batch_inputs)
                loss_one = model_one.loss_function(logits_one, batch_labels) 
            loss_one_sum += loss_one
            gradients_one = tape_one.gradient(loss_one, model_one.trainable_variables)
            model_one.optimizer.apply_gradients(zip(gradients_one, model_one.trainable_variables))
        avg_loss_one = loss_one_sum / count 
        print('training loss', avg_loss_one.numpy()) 
        total_loss += loss_one_sum 
    print('training loss avg:', total_loss/counter)
    return 
               

def test(model_one, test_input, test_labels):

    total_loss = 0
    counter = 0
    num_inputs = np.shape(test_labels)[0]
    num_days = np.shape(test_labels)[1]
    num_batches = num_days - num_days % model_one.batch_size
    for i, stock in enumerate(test_input):
        count = 0
        loss_one_sum = 0
        pointer = 0 
        while pointer < num_batches:
            count += 1
            counter += 1
            start = pointer
            pointer += model_one.batch_size
            batch_inputs = stock[start:pointer, :, :]
            batch_labels = test_labels[i, start:pointer]
            logits_one = model_one.call(batch_inputs)
            loss_one = model_one.loss_function(logits_one, batch_labels) 
            loss_one_sum += loss_one
        avg_loss_one = loss_one_sum / count 
        print('testing loss', avg_loss_one.numpy())
        total_loss += loss_one_sum 
    return total_loss/counter

def main():
    
    train_input, train_labels, test_input, test_labels = get_data("./data/prices.csv", "./data/fundamentals.csv")
    model_one = ModelOne()
    train(model_one, train_input, train_labels)
    avg_loss_one = test(model_one, test_input, test_labels)
    print('testing loss avg:', avg_loss_one)

       
if __name__ == '__main__':
    main()