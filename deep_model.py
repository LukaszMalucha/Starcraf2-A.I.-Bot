### NEURAL NETWORK

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import os
import random



model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(176, 200, 3),
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

learning_rate = 0.001
opt = keras.optimizers.adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/bot1")

training_data_dir = "train_data"

## Check the lengths of each choice list
def check_data():
    choices = {"marauders": marauders,
               "cyclones": cyclones,
               "thors": thors,
               "medivacs": medivacs}

    total_data = 0

    lengths = []
    for choice in choices:
        print("Length of {} is: {}".format(choice, len(choices[choice])))
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))

    print("Total data length now is:",total_data)
    return lengths

hm_epochs = 10

for i in range(hm_epochs):
    current = 0
    increment = 200  ## data chunk
    not_maximum = True
    all_files = os.listdir(training_data_dir) ## training files location
    maximum = len(all_files)
    random.shuffle(all_files)

    while not_maximum:
        print("WORKING ON {}:{}".format(current, current+increment))  ## current job
        marauders = []
        cyclones = []
        thors = []
        medivacs = []

        for file in all_files[current:current+increment]:         ## file iteration
            full_path = os.path.join(training_data_dir, file)
            data = np.load(full_path)                             ## load data
            data = list(data)                                     ## list data
            for d in data:                                        ## group choices
                choice = np.argmax(d[0])
                if choice == 0:
                    marauders.append([d[0], d[1]])
                elif choice == 1:
                    cyclones.append([d[0], d[1]])
                elif choice == 2:
                    thors.append([d[0], d[1]])
                elif choice == 3:
                    medivacs.append([d[0], d[1]])

        lengths = check_data()
        lowest_data = min(lengths)                              ## go to shortest choice list

        random.shuffle(marauders)
        random.shuffle(cyclones)
        random.shuffle(thors)
        random.shuffle(medivacs)

        marauders = marauders[:lowest_data]                           ## slice lists up to lowest data
        cyclones = cyclones[:lowest_data]
        thors = thors[:lowest_data]
        medivacs = medivacs[:lowest_data]

        check_data()


## COMBINE DATA

        training_data = marauders + cyclones + thors + medivacs

        random.shuffle(training_data)
        print(len(training_data))

        test_size = 100
        batch_size = 128

        ## reshape data to fit 
        x_train = np.array([i[1] for i in training_data[:-test_size]]).reshape(-1, 176, 200, 3)
        y_train = np.array([i[0] for i in training_data[:-test_size]])

        x_test = np.array([i[1] for i in training_data[-test_size:]]).reshape(-1, 176, 200, 3)
        y_test = np.array([i[0] for i in training_data[-test_size:]])

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  validation_data=(x_test, y_test),verbose=1,
                  shuffle=True, callbacks=[tensorboard])

        model.save("Apollyon_Terran")
        current += increment
        if current > maximum:
            not_maximum = False    