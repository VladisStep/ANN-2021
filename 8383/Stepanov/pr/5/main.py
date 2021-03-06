import numpy as np
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten

train_dataset = np.fromfile('train_dataset.csv', dtype ='float', sep =' ').reshape(-1, 2)
test_dataset = np.fromfile('test_dataset.csv', dtype ='float', sep =' ').reshape(-1, 2)

train_data = train_dataset[:, 0].reshape(-1,1)
train_targets = train_dataset[:, 1].reshape(-1,1)

test_data = test_dataset[:, 0].reshape(-1,1)
test_targets = test_dataset[:, 1].reshape(-1,1)

CONST_FOR_ENCODE = 5.0

inp = Input(shape=(1,))

# encoder
dense_ecoder = Dense(1, activation='relu', name='dense_ecoder')(inp)

# regression
dense_1 = Dense(64, activation='relu')(dense_ecoder)
dense_2 = Dense(64, activation='relu')(dense_1)
out_reg = Dense(1, name='out_reg')(dense_2)

# decoder
dense_decoder = Dense(1, name='dense_decoder')(dense_ecoder)

# configuration
model = Model(inputs=inp, outputs=[out_reg, dense_decoder, dense_ecoder])
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.fit(train_data, [train_targets, train_data, train_data * CONST_FOR_ENCODE], epochs=200, batch_size=10)


# encoder model
ecoder_model = Model(inputs=inp, outputs=dense_ecoder)
encoder_pred = np.asarray(ecoder_model.predict(test_data))
np.savetxt('outOfModels/data_encoder_model.csv', np.hstack((test_data * CONST_FOR_ENCODE, encoder_pred)))
ecoder_model.save('models/encoder_model.h5')

# regression model
reg_model = Model(inputs=inp, outputs=out_reg)
reg_pred = np.asarray(reg_model.predict(test_data))
np.savetxt('outOfModels/data_reg_model.csv', np.hstack((test_targets, reg_pred)))
ecoder_model.save('models/reg_model.h5')

# decoder model
decoder_model = Model(inputs=inp, outputs=dense_decoder)
decoder_pred = np.asarray(reg_model.predict(test_data * CONST_FOR_ENCODE))
np.savetxt('outOfModels/data_decoder_model.csv', np.hstack((test_data, decoder_pred)))
ecoder_model.save('models/decoder_model.h5')


print('end')