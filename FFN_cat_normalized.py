try:
    #This code was created with the help of ChatGPT3.5

    import tensorflow as tf
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt


    # Define the feedforward neural network model for binary classification
    def create_feedforward_nn(input_shape):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
            # tf.keras.layers.Dense(256, activation='relu'),
            # tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
        ])

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.003)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    # Load your dataset
    #----------------------------------------------------------------------------
    # 7 days before
    file_path = r'C:\Users\cn2802\Desktop\Junior IW\1. LIN_basic_material\1. LIN_LSTM_cont_processed_7.xlsx'
    #----------------------------------------------------------------------------
    # 7 days before EXTENDED
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\1. LIN_basic_material\1. LIN_LSTM_cont_processed_7_extended.xlsx'
    #----------------------------------------------------------------------------
    # 14 days before
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\1. LIN_basic_material\1. LIN_LSTM_cont_processed_14.xlsx'
    #----------------------------------------------------------------------------
    # 14 days before EXTENDED
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\1. LIN_basic_material\1. LIN_LSTM_cont_processed_14_extended.xlsx'
    #----------------------------------------------------------------------------
    # 21 days before
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_21.xlsx'
    #----------------------------------------------------------------------------
    # 21 days before EXTENDED
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_21_extended.xlsx'
    #----------------------------------------------------------------------------
    # 28 days before
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_28.xlsx'
    #----------------------------------------------------------------------------
    # 28 days before EXTENDED
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_28_extended.xlsx'
    #----------------------------------------------------------------------------
    # The 80_pct input
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\2. SHW_basic_material\2. SHW_80_pct.xlsx'
    #----------------------------------------------------------------------------
    #The 80_pct_input EXTENDED
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\2. SHW_basic_material\2. SHW_80_pct_extended.xlsx'
    #----------------------------------------------------------------------------
    #Research_paper_input
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\2. SHW_basic_material\2. SHW_Research_paper_input.xlsx'
    #----------------------------------------------------------------------------
    #Research_paper_input EXTENDED
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\2. SHW_basic_material\2. SHW_Research_paper_input_extended.xlsx'
    #----------------------------------------------------------------------------
    #The 80_pct_input extended (3 companies)
    # file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\80_pct_acc_input_features_categorical_extended_3_com.xlsx'
    #----------------------------------------------------------------------------
    #Basic features input
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\2. SHW_basic_material\2. SHW_Basic_features.xlsx'
    #----------------------------------------------------------------------------
    #Basic features input EXTENDED
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\2. SHW_basic_material\2. SHW_Basic_features_extended.xlsx'


    df = pd.read_excel(file_path)
    # Convert feature names to strings
    df.columns = df.columns.astype(str)

    # Separate features (X) and target variable (y)
    X = df.iloc[:, :-1]  # Exclude the last column which is the target variable
    y = df.iloc[:, -1]   # The last column is the target variable

    # Normalize the input features using Min-Max scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Determine the sizes of the training, validation, and test sets
    train_size = int(0.7 * len(df))  # 70% of the data for training
    val_size = int(0.15 * len(df))    # 15% of the data for validation
    test_size = len(df) - train_size - val_size  # Remaining data for test

    # Split the data into training, validation, and test sets based on time
    X_train = X_scaled[:train_size]
    y_train = y[:train_size]
    X_val = X_scaled[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X_scaled[train_size + val_size:]
    y_test = y[train_size + val_size:]

    #Convert y to binary of 0 and 1
    y_train[y_train == -1] = 0
    y_val[y_val == -1] = 0
    y_test[y_test == -1] = 0

    # Create the feedforward neural network model
    input_shape = X_train.shape[1:]  # Input shape is the number of features
    feedforward_nn_model = create_feedforward_nn(input_shape)

    # # Train the model
    # history = feedforward_nn_model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val))

    # # Evaluate the model on the training set
    # train_loss, train_accuracy = feedforward_nn_model.evaluate(X_train, y_train)

    # # Evaluate the model on the validation set
    # val_loss, val_accuracy = feedforward_nn_model.evaluate(X_val, y_val)

    # # Perform predictions on test data
    # y_pred = feedforward_nn_model.predict(X_test)

    # # Evaluate the model on the test set
    # test_loss, test_accuracy = feedforward_nn_model.evaluate(X_test, y_test)

    # # Define a callback to track the average validation accuracy over the past 10 epochs
    # class TrackValidationAccuracy(tf.keras.callbacks.Callback):
    #     def __init__(self):
    #         super(TrackValidationAccuracy, self).__init__()
    #         self.validation_accuracies = []

    #     def on_epoch_end(self, epoch, logs=None):
    #         self.validation_accuracies.append(logs['val_accuracy'])

    # # Instantiate the callback
    # track_validation_accuracy = TrackValidationAccuracy()

    # # Train the model with the callback
    # history = feedforward_nn_model.fit(X_train, y_train, epochs=500, batch_size=128,
    #                                    validation_data=(X_val, y_val), callbacks=[track_validation_accuracy])

    # # Calculate the average validation accuracy over the past 10 epochs
    # avg_validation_accuracies = [sum(track_validation_accuracy.validation_accuracies[i:i+10]) / 10
    #                               for i in range(len(track_validation_accuracy.validation_accuracies) - 9)]

    # # Find the epoch with the highest average validation accuracy
    # best_epoch = avg_validation_accuracies.index(max(avg_validation_accuracies)) + 10  # Add 10 to get the actual epoch

    # print("Best Epoch based on Average Validation Accuracy:", best_epoch)

    # # Re-train the model with the best epoch
    # feedforward_nn_model.fit(X_train, y_train, epochs=best_epoch, batch_size=128, validation_data=(X_val, y_val))

    # # Evaluate the model on the test set
    # test_loss, test_accuracy = feedforward_nn_model.evaluate(X_test, y_test)

    # print("Test Accuracy after re-training:", test_accuracy)
    # print("Test Loss after re-training:", test_loss)

    # Define a callback to save the model with the highest validation accuracy
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Train the model
    history = feedforward_nn_model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_data=(X_val, y_val), callbacks=[checkpoint_callback])

    # Load the best model
    best_model = tf.keras.models.load_model("best_model.keras")

    # Evaluate the best model on the test set
    test_loss, test_accuracy = best_model.evaluate(X_test, y_test)

    # Print the validation accuracy of the best model
    val_loss, val_accuracy = best_model.evaluate(X_val, y_val)
    

    # Print the training accuracy of the best model
    train_loss, train_accuracy = best_model.evaluate(X_train, y_train)
    print("Training Accuracy of Best Model:", train_accuracy)
    print("Validation Accuracy of Best Model:", val_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("Test Loss:", test_loss)

    # # Invert the normalization of predictions
    # y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

    # # Create a DataFrame to store actual and predicted data for each data point
    # results_df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})

    # # Save the DataFrame to an Excel file
    # results_df.to_excel('results_FFN_pct_cat.xlsx', index=False)

    # Plot validation loss and validation accuracy
    plt.figure(figsize=(10, 5))

    # Plot validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss vs. Epoch')
    plt.legend()
    plt.grid(True)

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs. Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
except Exception as exception:
    # print(exception)
    raise(exception)
