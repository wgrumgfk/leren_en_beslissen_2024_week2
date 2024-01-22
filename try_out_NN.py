import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load your dataset (replace 'your_dataset.csv' with the actual file name)
df = pd.read_csv('/home/bardha/Documents/GitHub/leren_en_beslissen_2024_week2/okeanos_processed.csv')

selected_columns = ['man', '500_split_sec', 'zwaar', 'licht', 'AT', 'I', 'ID', 'ED', 'ED+', 'ervaring']


# Filter rows with missing values in selected columns
df_filtered = df[selected_columns + ['two_k_tijd_sec']].dropna()

# Separate features and target variable
features = df_filtered.drop('two_k_tijd_sec', axis=1) 
target = df_filtered['two_k_tijd_sec']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
print(y_pred)
print(y_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')