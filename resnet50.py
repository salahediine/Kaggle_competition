resnet_model =  tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='max',  
                                              input_shape=(299,299,3))

model = models.Sequential()
model.add(resnet_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(train['category_id'].unique().shape[0], activation='softmax'))  



resnet_train_datagen = ImageDataGenerator(
featurewise_center=False,
 featurewise_std_normalization=False,
 rotation_range=180,
 width_shift_range=0.1,
 height_shift_range=0.1,
 zoom_range=0.2,
preprocessing_function =tf.keras.applications.resnet.preprocess_input)



resnet_test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input)

hist_resnet = model.fit_generator(resnet_train_datagen.flow_from_dataframe(dataframe=train,
                                                      directory=train_dir,
                                                      x_col="file_name",
                                                      y_col="category_id",
                                                      target_size=(299, 299),
                                                      batch_size=32,
                                                     class_mode='categorical'),
                    validation_data=resnet_train_datagen.flow_from_dataframe(
                        dataframe=test,
                        directory= train_dir,
                        x_col="file_name",
                        y_col="category_id",
                        target_size=(299, 299),
                        batch_size=32,
                    class_mode='categorical'),
                    epochs=10,
                    steps_per_epoch=len(train)//32,
                    validation_steps=len(test)//32,
                    verbose=2)