import nltk
import joblib

# Load the pre-trained model
model = joblib.load('suicidal_text_detection_model.pkl')

# Define a function to preprocess the input text
def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    # Perform stemming
    stemmer = nltk.stem.PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    # Join the tokens back into a string
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text

# Define a function to detect suicidal text input
def detect_suicidal_text(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    # Use the pre-trained model to make a prediction
    prediction = model.predict([preprocessed_text])[0]
    # Return the prediction as a boolean value
    return bool(prediction)

# Define a function to handle user input and generate a response
def handle_user_input(user_input):
    response = ''
    # Check if the user input contains suicidal text
    if detect_suicidal_text(user_input):
        response = 'I am sorry to hear that, You seems to have a suicidal thought.'
    else:
        response = 'I am here to help you. How can I assist you today?'
    return response

# Define a function to start the chatbot interaction
def start_chatbot():
    print('Hi, I am your mental health chatbot. How can I help you today?')
    while True:
        user_input = input('You: ')
        # Check if the user wants to exit the chatbot interaction
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print('Chatbot: Thank you for using the mental health chatbot. Take care!')
            break
        # Generate a response based on the user input
        response = handle_user_input(user_input)
        print('Chatbot:', response)

# Start the chatbot interaction
start_chatbot()
