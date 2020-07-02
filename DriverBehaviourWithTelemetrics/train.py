# import libraries
import numpy as np
import tensorflow as tf

def train():
    # load datasets
    x = np.load('preprocessed_data\\input.npy')
    y = np.load('preprocessed_data\\output.npy').astype('int32')
    print(np.sum(y))
    print(y.shape[0])
    print('Num of positives = ', np.sum(y))
    print('Num of negatives = ', y.shape[0]-np.sum(y))

    # shuffle datasets
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    # split datasets
    x_train = x[0:9000, :, :]
    y_train = y[0:9000, :]
    x_test = x[9001:, :, :]
    y_test = y[9001:, :]
    print('Num of positives = ', np.sum(y_train))
    print('Num of negatives = ', y_train.shape[0]-np.sum(y_train))
    print('Num of positives = ', np.sum(y_test))
    print('Num of negatives = ', y_test.shape[0]-np.sum(y_test))

    # convert output to one-hot encoding
    num_class = 2
    y_train = tf.keras.utils.to_categorical(y_train,num_classes=num_class)
    y_test = tf.keras.utils.to_categorical(y_test,num_classes=num_class)

    # define LSTM model
    model =  tf.keras.models.Sequential([
        tf.keras.layers.LSTM(100, input_shape=(x_train.shape[1],x_train.shape[2])),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(num_class, activation='softmax')
    ])

    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train model
    model.fit(x_train, y_train, epochs=800, validation_split=0.1)
    model.save('model\\lstm_downsampled.h5')

    # evaluate model
    score_test = model.evaluate(x_test, y_test, verbose=0)
    print('Test Error: %.4f' % score_test[0])
    print('Test Accuracy: %.4f' % score_test[1])

if __name__ == "__main__":
    train()