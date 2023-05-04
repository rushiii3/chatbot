# Importing necessary libraries
import nltk
import numpy as np
import tensorflow as tf
import random
import json

# Downloading necessary NLTK data
nltk.download('punkt')

# Loading and parsing the data
with open('intents.json') as file:
    data = json.load(file)

# Tokenizing the patterns in the data
words = []
labels = []
patterns = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize the pattern into individual words
        tokenized_words = nltk.word_tokenize(pattern)
        words.extend(tokenized_words)
        patterns.append(tokenized_words)
        labels.append(intent['tag'])
        
    # Add the responses for each intent
    responses.append(intent['responses'])

# Remove duplicates and sort the words
words = sorted(list(set(words)))

# Create a dictionary of words and their corresponding index
word_indices = {}
for i, word in enumerate(words):
    word_indices[word] = i

# Create the training data
training_data = []
output_data = []

for i, pattern in enumerate(patterns):
    bag = []
    
    # Create a bag of words for each pattern
    for word in words:
        if word in pattern:
            bag.append(1)
        else:
            bag.append(0)
    
    # Create the output data for the pattern
    output = [0] * len(labels)
    output[labels.index(labels[i])] = 1
    
    training_data.append(bag)
    output_data.append(output)

# Convert the data to numpy arrays
training_data = np.array(training_data)
output_data = np.array(output_data)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(training_data[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(output_data[0]), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(training_data, output_data, epochs=500, batch_size=8)

# Save the model
model.save('chatbot_model.h5')

# Load the model
model = tf.keras.models.load_model('chatbot_model.h5')

# Define a function to generate responses
def generate_response(user_input):
    # Tokenize the user input
    tokenized_user_input = nltk.word_tokenize(user_input)
    
    # Create a bag of words for the user input
    bag = [0] * len(words)
    for word in tokenized_user_input:
        if word in words:
            bag[word_indices[word]] = 1
            
    # Predict the intent for the user input
    prediction = model.predict(np.array([bag]))
    intent_index = np.argmax(prediction)
    
    # Choose a random response for the predicted intent
    return random.choice(responses[intent_index])

# Main loop for the chatbot
while True:
    user_input = input("You: ")
    chatbot_response = generate_response(user_input)
    print("Chatbot:", chatbot_response)
