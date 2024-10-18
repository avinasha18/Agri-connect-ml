# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset from the CSV file
data = pd.read_csv('crop.csv')  # Ensure this is the correct path to your file

# Features (X) and target (y)
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]  # Features
y = data['label']  # Target label (crop name)

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForestClassifier
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_train, y_train)

# Function to recommend top N crops based on probability
def recommend_top_crops(model, X_test, top_n=3):
    # Predict probabilities for each class (crop)
    probabilities = model.predict_proba(X_test)
    
    # Get the class (crop) labels
    crop_labels = model.classes_
    
    # For each test sample, get the top N crops with the highest probabilities
    top_crops = []
    for i in range(len(probabilities)):
        # Get the indices of the top N probabilities
        top_indices = probabilities[i].argsort()[-top_n:][::-1]
        # Get the corresponding crops
        top_crops.append(crop_labels[top_indices])
    
    return top_crops

# Use the updated function to recommend top 3 crops for the test set
top_crops = recommend_top_crops(crop_model, X_test, top_n=3)

# Display the recommendations for the first few test samples
for i in range(5):
    print(f"Top 3 crop recommendations for sample {i+1}: {top_crops[i]}")

joblib.dump(crop_model,"crop_recommendation_five.joblib")