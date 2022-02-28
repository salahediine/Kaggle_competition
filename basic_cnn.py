model = Sequential()

#1 block
model.add(Conv2D(filters=16,kernel_size=(3,3), padding='same',use_bias=False,input_shape=(299,299,3),
         activation = 'relu'))
model.add(Conv2D(filters=16,kernel_size=(3,3), padding='same',use_bias=False,input_shape=(299,299,3),
         activation = 'relu'))
model.add(BatchNormalization(scale=False))
model.add(MaxPooling2D(pool_size=(4, 4),strides=(4, 4),padding='same'))
model.add(Dropout(0.25))

# 2end block
model.add(Conv2D(filters=32,kernel_size=(3,3), padding='same',use_bias=False, activation ='relu'))
model.add(Conv2D(filters=32,kernel_size=(3,3), padding='same',use_bias=False, activation ='relu'))
model.add(BatchNormalization(scale=False))
model.add(MaxPooling2D(pool_size=(4, 4),strides=(4, 4),padding='same'))
model.add(Dropout(0.25))

# 3d block
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', use_bias=False, activation ='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', use_bias=False, activation ='relu'))
model.add(BatchNormalization(axis=3, scale=False))
model.add(Flatten())


model.add(Dense(128, activation='relu'))
model.add(Dense(train['category_id'].unique().shape[0], activation = 'softmax'))