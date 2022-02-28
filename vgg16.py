from keras.applications.vgg16 import VGG16 ### VGG16
imnet = VGG16(weights="imagenet", include_top=False, input_shape=(299, 299, 3))

model = models.Sequential() 
model.add(imnet) 
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(train['category_id'].unique().shape[0], activation='softmax')) 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m]) 

imnet_train_datagen = ImageDataGenerator(
featurewise_center=False,
 featurewise_std_normalization=False,
 rotation_range=180,
 width_shift_range=0.1,
 height_shift_range=0.1,
 zoom_range=0.2,
preprocessing_function =tf.keras.applications.vgg16.preprocess_input)