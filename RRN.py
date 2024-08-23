#This code was created with the help of ChatGPT3.5
try:

    import tensorflow as tf
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import numpy as np


    # Define the recurrent neural network model for binary classification
    def create_rnn(input_shape):
        model = tf.keras.models.Sequential([
            tf.keras.layers.SimpleRNN(256, activation='relu', return_sequences=True, input_shape=input_shape),
            # tf.keras.layers.SimpleRNN(256, activation='relu', return_sequences=True),
            tf.keras.layers.SimpleRNN(128, activation='relu', return_sequences=True),
            tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=True),
            tf.keras.layers.SimpleRNN(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
        ])

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.003)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    # Load your dataset
    #----------------------------------------------------------------------------
    # 7 days before
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_7.xlsx'
    #----------------------------------------------------------------------------
    # 7 days before EXTENDED
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_7_extended.xlsx'
    #----------------------------------------------------------------------------
    # 14 days before
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_14.xlsx'
    #----------------------------------------------------------------------------
    # 14 days before EXTENDED
    # file_path = r'C:\Users\cn2802\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_14_extended.xlsx'
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
    file_path = r'C:\Users\cn2802\Desktop\Junior IW\2. SHW_basic_material\2. SHW_Basic_features_extended.xlsx'


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

    X_train = tf.expand_dims(X_train, axis=-1)
    y_train = tf.expand_dims(y_train, axis=-1)
    X_val = tf.expand_dims(X_val, axis=-1)
    y_val = tf.expand_dims(y_val, axis=-1)
    X_test = tf.expand_dims(X_test, axis=-1)
    y_test = tf.expand_dims(y_test, axis=-1)

    # Convert y to binary of 0 and 1
    y_train = tf.where(y_train == -1, 0, y_train)
    y_val = tf.where(y_val == -1, 0, y_val)
    y_test = tf.where(y_test == -1, 0, y_test)

    # Create the feedforward neural network model
    input_shape = X_train.shape[1:]  # Input shape is the number of features
    recurrent_nn_model = create_rnn(input_shape)

    # Define a callback to save the model with the highest validation accuracy
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Train the model
    history = recurrent_nn_model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val), callbacks=[checkpoint_callback])

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

    # # Convert y_test EagerTensor to NumPy array
    # y_test_np = y_test.numpy()

    # # Create a DataFrame to store actual and predicted data for each data point
    # results_df = pd.DataFrame({'Actual': y_test_np.flatten(), 'Predicted': y_pred.flatten()})

    # # Save the DataFrame to an Excel file
    # results_df.to_excel('results_RNN_7_days.xlsx', index=False)

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
