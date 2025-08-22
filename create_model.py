# create_model.py
import pandas as pd
import pickle
from model import SimpleNaiveBayes

# Load your dataset
df = pd.read_csv('osteoporosis.csv')

# Replace 'target_column' with the actual target column name in your dataset
X = df.drop('Osteoporosis', axis=1)  # Adjust 'Osteoporosis' to match your target column
y = df['Osteoporosis']

# Train the model
model = SimpleNaiveBayes()
model.fit(X, y)

# Save the model to a new pickle file
with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(model, f)