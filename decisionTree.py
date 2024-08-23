import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

try:
    # Load your dataset
    #----------------------------------------------------------------------------
    # 7 days before
    # file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_7.xlsx'
    #----------------------------------------------------------------------------
    # 7 days before EXTENDED
    # file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_7_extended.xlsx'
    #----------------------------------------------------------------------------
    # 14 days before
    # file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_14.xlsx'
    #----------------------------------------------------------------------------
    # 14 days before EXTENDED
    # file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_14_extended.xlsx'
    #----------------------------------------------------------------------------
    # 21 days before
    # file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_21.xlsx'
    #----------------------------------------------------------------------------
    # 21 days before EXTENDED
    # file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_21_extended.xlsx'
    #----------------------------------------------------------------------------
    # 28 days before
    # file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_28.xlsx'
    #----------------------------------------------------------------------------
    # 28 days before EXTENDED
    # file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\2. SHW_basic_material\2. SHW_LSTM_cont_processed_28_extended.xlsx'
    #----------------------------------------------------------------------------
    # The 80_pct input
    # file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\2. SHW_basic_material\2. SHW_80_pct.xlsx'
    #----------------------------------------------------------------------------
    #The 80_pct_input EXTENDED
    # file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\2. SHW_basic_material\2. SHW_80_pct_extended.xlsx'
    #----------------------------------------------------------------------------
    #Research_paper_input
    # file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\2. SHW_basic_material\2. SHW_Research_paper_input.xlsx'
    #----------------------------------------------------------------------------
    #Research_paper_input EXTENDED
    # file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\2. SHW_basic_material\2. SHW_Research_paper_input_extended.xlsx'
    #----------------------------------------------------------------------------
    #The 80_pct_input extended (3 companies)
    # file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\80_pct_acc_input_features_categorical_extended_3_com.xlsx'
    #----------------------------------------------------------------------------
    #Basic features input
    # file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\2. SHW_basic_material\2. SHW_Basic_features.xlsx'
    #----------------------------------------------------------------------------
    #Basic features input EXTENDED
    file_path = r'C:\Users\nguye\OneDrive\Desktop\Junior IW\2. SHW_basic_material\2. SHW_Basic_features_extended.xlsx'

    df = pd.read_excel(file_path)

    # Convert feature names to strings
    df.columns = df.columns.astype(str)

    # Separate features (X) and target variable (y)
    X = df.iloc[:, :-1]  # Exclude the last column which is the target variable
    y = df.iloc[:, -1]   # The last column is the target variable

    # #-------------- DATA SPLIT FOR NON TIME SERIES DATA ---------------------------------------------
    # # Split the data into training, validation, and test sets, USE FOR NON TIME SERIES DATA
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    #-------------- DATA SPLIT FOR TIME SERIES DATA ---------------------------------------------------
    # Determine the sizes of the training, validation, and test sets
    train_size = int(0.7 * len(df))  # 70% of the data for training
    val_size = int(0.15 * len(df))    # 15% of the data for validation
    test_size = len(df) - train_size - val_size  # Remaining data for test

    # Split the data into training, validation, and test sets
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    # Define a range of max depths to try
    max_depth_values = range(1, 20)  # Values from 1 to 100

    # Initialize variables to store the best max depth and its corresponding accuracy
    best_max_depth = None
    best_accuracy = 0

    # Initialize lists to store train and test accuracies
    val_accuracies = []
    
    # Iterate over different max depths
    for max_depth in max_depth_values:
        # Initialize the Decision Tree Classifier with the current max depth
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

        # Train the model on the training data
        model.fit(X_train, y_train)

        # Make predictions on the validation set
        val_predictions = model.predict(X_val)

        # Calculate the accuracy on the validation set
        accuracy = accuracy_score(y_val, val_predictions)
        val_accuracies.append(accuracy)

        # Check if this model has the best accuracy so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_max_depth = max_depth

    # print(val_accuracies)
    print("Best Max Depth:", best_max_depth)

    # Retrain the best model on the entire training dataset
    best_model = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
    best_model.fit(X_train, y_train)

    # Predict labels for training data
    train_predictions = model.predict(X_train)
    # Calculate training accuracy
    train_accuracy = accuracy_score(y_train, train_predictions)

    # Predict labels for validation data
    val_predictions = model.predict(X_val)
    # Calculate validation accuracy
    val_accuracy = accuracy_score(y_val, val_predictions)

    print("Training Accuracy:", train_accuracy)
    print("Validation Accuracy:", val_accuracy)

    # Make final predictions on the test set
    test_predictions = best_model.predict(X_test)

    # Calculate the accuracy of the final predictions
    final_accuracy = accuracy_score(y_test, test_predictions)

    print("Final Test Accuracy:", final_accuracy)

    # Create a DataFrame to store the actual data and the final predictions
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': test_predictions})

    # Save the DataFrame to an Excel file
    # results_df.to_excel('DT_validated_28_days.xlsx', index=False)

    # Make a plot of different max depths and corresponding accuracies
    plt.plot(max_depth_values, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracies vs. Max Depth')
    plt.legend()
    plt.grid(True)
    plt.show()

except Exception as exception:
    print(exception)
    raise(exception)
