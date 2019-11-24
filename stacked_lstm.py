import tensorflow as tf
import numpy as np
from preprocess import get_data
import math

class Model(tf.keras.Model):

    def __init__(self):

        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.batch_size = 10
        self.rnn_size = 128
        self.first_lstm = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=True, return_state=True)
        self.dense_layer = tf.keras.layers.Dense(units=1762, activation='softmax')
       
    def call(self, inputs):
   
        output_seq, final_out, final_state = self.first_lstm(inputs)
        return self.dense_layer(output_seq)
        
    def loss(self, logits, labels):
        
        return tf.math.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))

def train(model, train_input, train_labels):
    
    num_inputs = np.shape(train_labels)[0]
    pointer = 0 
    while pointer < num_inputs:
        start = pointer
        pointer += model.batch_size
        batch_inputs = train_input[start:pointer, :]
        batch_labels = train_labels[start:pointer, :]
        with tf.GradientTape() as tape:
            batch_logits = model.call(batch_inputs)
            loss = model.loss(batch_logits, batch_labels)
            print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return 

def test(model, test_inputs, test_labels):
    
    pass

def main():
    
    train_input, train_labels, test_input, test_labels = get_data("./data/prices.csv", "./data/fundamentals.csv")

    print(train_input.shape)
    print(train_labels.shape)
    print(test_input.shape)
    print(test_labels.shape)

    model = Model()
    train(model, train_input, train_labels)

    
if __name__ == '__main__':
    main()
