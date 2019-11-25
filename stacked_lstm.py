import tensorflow as tf
import numpy as np
from preprocess import get_data
import math

class Model(tf.keras.Model):

    def __init__(self):

        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.batch_size = 100
        self.window_size = 21
        self.rnn_size = 256

        # normal implem
        self.lstm_one = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=True, return_state=True, stateful=True)
        self.dropout_one = tf.keras.layers.Dropout(0.3)
        self.lstm_two = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=False, return_state=True)
        self.dropout_two = tf.keras.layers.Dropout(0.3)
        self.dense_layer_one = tf.keras.layers.Dense(units=32, kernel_initializer="uniform", activation='relu')
        self.dense_layer_two = tf.keras.layers.Dense(units=1, kernel_initializer="uniform", activation='linear')

        # stacked layer 
        self.lstm_one_s = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=True, return_state=True, stateful=True)
        self.dropout_one_s = tf.keras.layers.Dropout(0.3)
        self.lstm_two_s = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=False, return_state=True)
        self.dropout_two_s = tf.keras.layers.Dropout(0.3)
        self.dense_layer_one_s = tf.keras.layers.Dense(units=32, kernel_initializer="uniform", activation='relu')
        self.dense_layer_two_s = tf.keras.layers.Dense(units=1, kernel_initializer="uniform", activation='linear')
       
    def call(self, inputs):
        output_seq, final_out, final_state = self.lstm_one(inputs)
        output_seq = self.dropout_one(output_seq)
        a, final_out, c = self.lstm_two(output_seq)
        final_out = self.dropout_two(final_out)
        dense_one_out = self.dense_layer_one(final_out)
        return self.dense_layer_two(dense_one_out)
    
    def call_stack(self, inputs):
        output_seq, final_out, final_state = self.lstm_one_s(inputs)
        output_seq = self.dropout_one_s(output_seq)
        a, final_out, c = self.lstm_two_s(output_seq)
        final_out = self.dropout_two_s(final_out)
        dense_one_out = self.dense_layer_one_s(final_out)
        return self.dense_layer_two_s(dense_one_out)
        
    def loss(self, logits, labels):
      
        return tf.math.reduce_mean(tf.keras.losses.MSE(labels, logits))

def train(model, train_input, train_labels):
    
    num_inputs = np.shape(train_labels)[0]
    num_days = np.shape(train_labels)[1]
    num_batches = num_days - num_days % model.batch_size
    counter = 0
    for i, stock in enumerate(train_input):
        pointer = 0 
        while pointer < num_batches:

            start = pointer
            pointer += model.batch_size
            batch_inputs = stock[start:pointer, :, :]
            batch_labels = train_labels[i, start:pointer]

            with tf.GradientTape() as tape:

                ############# NEW IMPLEM #####################

                # normal 
                batch_logits = np.array(model.call(batch_inputs))
                
                # make new dataset of predictions and real labels 
                batch_labels = np.reshape(batch_labels,(100, -1))
                new_data = np.hstack((batch_logits, batch_labels)) # shape = (100, 2)
                predictions = []
                window = 9
                for j in range(model.batch_size - window): 
                    predictions.append(new_data[j:j + window]) 
                predictions = np.array(predictions) # shape = (91, 9, 2) 
                new_labels = batch_labels[window:] # shape = (91, 1)
                new_logits = model.call_stack(predictions)

                loss = model.loss(new_logits, new_labels)

                ################ ORIGINAL ##########################
                 
                # batch_logits = np.array(model.call(batch_inputs))
                # loss = model.loss(batch_logits, batch_labels) 

                print(loss)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return 

def test(model, test_inputs, test_labels):
    
    pass

def main():
    
    train_input, train_labels, test_input, test_labels = get_data("./data/prices.csv", "./data/fundamentals.csv")

    # print(train_input.shape)
    # print(train_labels.shape)
    # print(test_input.shape)
    # print(test_labels.shape)

    model = Model()
    train(model, train_input, train_labels)

    
if __name__ == '__main__':
    main()
