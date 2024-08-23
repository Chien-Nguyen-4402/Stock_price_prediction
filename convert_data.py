# This was created with the help of ChatGPT3.5
# Stock prices are formatted in columns. This code converts the data to
# row formate to make it more convenient for ML training
import pandas as pd

# Read data from Excel file
df = pd.read_excel('LSTM_cont_raw.xlsx', header=None, names=['A'])
# print(df)
# Create a list to store the rows
rows = []

# Iterate over the values in column A
for i in range(len(df) - 29):
    # Extract values from Ai to A(i+22)
    row_values = df['A'].iloc[i:i+30].tolist()
    # Append the row to the list
    # print(row_values)
    rows.append(row_values)

# Create a new DataFrame with the rows
new_df = pd.DataFrame(rows)

# If you want to fill NaN values with 0:
new_df = new_df.fillna(0)

# If you want to drop rows with NaN values:
# new_df = new_df.dropna()

# Print the new DataFrame
# print(new_df)
# Save the new DataFrame to an Excel file
new_df.to_excel('4. META_LSTM_cont_processed_28.xlsx', index=False)