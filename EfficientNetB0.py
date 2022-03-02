from tensorflow.keras.applications import EfficientNetB0

efficientnet = EfficientNetB0(include_top=True,
                              weights=None,
                              input_shape=(299, 299, 3),
                              classes=train['category_id'].unique().shape[0])

efficientnet.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy', f1_m])

efficientnet_datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=180,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

hist_effnet = efficientnet.fit_generator(
    efficientnet_datagen.flow_from_dataframe(dataframe=train,
                                             directory=train_dir,
                                             x_col="file_name",
                                             y_col="category_id",
                                             target_size=(299, 299),
                                             batch_size=32,
                                             class_mode='categorical'),
    validation_data=efficientnet_datagen.flow_from_dataframe(
        dataframe=test,
        directory=train_dir,
        x_col="file_name",
        y_col="category_id",
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical'),
    epochs=10,
    steps_per_epoch=len(train) // 32,
    validation_steps=len(test) // 32,
    verbose=2)