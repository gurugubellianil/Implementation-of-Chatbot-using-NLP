import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Initialize the session state for chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

counter = 0

def main():
    global counter
    st.title("Chatbot with NLP")

    # Sidebar menu options
    menu = ["Chat", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Chat Menu
    if choice == "Chat":
        st.write("Feel free to chat with the bot! Type your message below:")

        # Check if chat_log.csv exists, create it if not
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        # Input field placed at the top
        user_input = st.text_input("You:", key=f"user_input_{counter}", placeholder="Type your message here...")

        if user_input:
            response = chatbot(user_input)

            # Append the new interaction to chat history immediately
            st.session_state['history'].append({'user': user_input, 'bot': response})

            # Log chat to CSV
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            # Clear the input box for new messages
            counter += 1

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

        # Display the conversation history in reverse order (newest message first)
        st.write("**Chat History**")
        for chat in reversed(st.session_state['history']):
            st.markdown(f"<div style='text-align:left; color:red;'><strong>You:</strong> {chat['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:right; color:green;'><strong>Bot:</strong> {chat['bot']}</div>", unsafe_allow_html=True)
            st.markdown("---")

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history available.")

    # About Menu
    elif choice == "About":
        st.write("This is a chatbot built using NLP techniques with a Logistic Regression classifier.")
        st.write("The project is developed using Streamlit, providing a user-friendly web interface for conversation.")

if __name__ == '__main__':
    main()
