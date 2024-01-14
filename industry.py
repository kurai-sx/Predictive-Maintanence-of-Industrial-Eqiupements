import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np

dataset = pd.read_csv("predictive_maintenance.csv")
print("<dataset imported>".upper())
# Preprocess the categorical 'Type' column
dataset['Type'] = dataset['Type'].map({'H': 2, 'L': 0, 'M': 1})
print("<type['L','M','H'] ===>> type[0,1,2]>")
# Separation
X = dataset[['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y = dataset['Failure Type']

dataset.head()

dataset.info()



# Convert categorical 'Failure Type' to numerical labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
print(y[0])
y = label_encoder.fit_transform(y)
print("output converted to numeric".upper())

# Build the model
model = Sequential()
af = "relu"
model.add(Input(shape=(X.shape[1],)))
model.add(Dense(128, activation=af))
model.add(Dense(64, activation=af))
model.add(Dense(32, activation=af))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer with softmax for multi-class classification

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=100, batch_size=64, validation_split=0.2)

for i in range(100):
    #if dataset.loc[i,"Failure Type"] == "Heat Dissipation Failure":

        print("=====================================================================================")

        print("Input:",X.iloc[[i]])
        print("Output:",model.predict(X.iloc[[i]]))
        print("Failure Type:",dataset.loc[i,"Failure Type"])

from sklearn.metrics import accuracy_score, precision_score
# Make predictions
y_probs = model.predict(X)
y_pred = y_probs.argmax(axis=-1)

# Evaluate accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Evaluate precision
precision = precision_score(y, y_pred, average='weighted')
print(f"Precision: {precision:.2f}")



model.save('model.h5')

X.head()

inps = np.array([1, 298, 300, 1230, 43, 1])
inps.resize(1, 6)

c = model.predict(f)
def answer(l):
    l = np.array(l) 
    l.resize(1, 6)
    c = model.predict(l)
    return c.max()
