import tensorflow as tf
import numpy as np
from preprocess import get_data
import math
import matplotlib.pyplot as plt

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
    
    train_loss_one = []
    train_loss_two = []

    counter = 0
    total_loss_one = 0
    total_loss_two = 0
    num_inputs = np.shape(train_labels)[0]
    num_days = np.shape(train_labels)[1]
    num_batches = num_days - num_days % model_one.batch_size
    for i, stock in enumerate(train_input):
        count = 0
        pointer = 0 
        loss_one_sum = 0
        loss_two_sum = 0
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

            loss_one_sum += loss_one
            loss_two_sum += loss_two
            total_loss_one += loss_one
            total_loss_two += loss_two

            gradients_one = tape_one.gradient(loss_one, model_one.trainable_variables)
            model_one.optimizer.apply_gradients(zip(gradients_one, model_one.trainable_variables))
            gradients_two = tape_two.gradient(loss_two, model_two.trainable_variables)
            model_two.optimizer.apply_gradients(zip(gradients_two, model_two.trainable_variables))

        if i % 20 == 0:
            train_loss_one.append((total_loss_one/counter).numpy())
            train_loss_two.append((total_loss_two/counter).numpy())
        
        # avg_loss_one = loss_one_sum/count
        # avg_loss_two = loss_two_sum/count
        # print('loss one', avg_loss_one.numpy())
        # print('loss two', avg_loss_two.numpy())

    print('training loss one avg:', total_loss_one/counter)
    print('training loss two avg:', total_loss_two/counter)
    return train_loss_one, train_loss_two
               

def test(model_one, model_two, test_input, test_labels):

    test_loss_one = []
    test_loss_two = []
    
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
        if i % 20 == 0:
            test_loss_one.append((total_loss_one/counter).numpy())
            test_loss_two.append((total_loss_two/counter).numpy())

    return test_loss_one, test_loss_two, total_loss_one/counter, total_loss_two/counter

def visualize_results(tr1, tr2, ts1, ts2):
    epoch_count = range(1, len(tr1) + 1)
    plt.plot(epoch_count, tr1, 'r-')
    plt.plot(epoch_count, ts1, 'b--')
    plt.plot(epoch_count, tr2, 'g-')
    plt.plot(epoch_count, ts2, 'p--')
    plt.legend(['Training Loss One', 'Test Loss One', 'Training Loss Two', 'Test Loss Two'])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

def main():
    
    train_input, train_labels, test_input, test_labels = get_data("./data/prices.csv", "./data/fundamentals.csv")
    model_one = ModelOne()
    model_two = ModelTwo()
    # num_epochs = 7

    # for _ in range(num_epochs):
    tr1, tr2 = train(model_one, model_two, train_input, train_labels)
    ts1, ts2, avg_loss_one, avg_loss_two = test(model_one, model_two, test_input, test_labels)
    print('testing loss one avg:', avg_loss_one.numpy())
    print('testing loss two avg:', avg_loss_two.numpy())

    visualize_results(tr1, tr2, ts1, ts2)
    
    
if __name__ == '__main__':
    main()


