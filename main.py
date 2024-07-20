+import speech_recognition as sr
import pyttsx3
import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            speak("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Sorry, my speech service is down.")
            speak("Sorry, my speech service is down.")
            return None

# Sample FAQ data
faq_data = {
    "What is your name?": "I am an AI-based chatbot.",
    "How can you help me?": "I can assist you with your queries.",
    "What services do you offer?": "I offer various services including information retrieval and task automation.",
    "How to contact support?": "You can contact support via email at support@example.com."
}

# Convert FAQ data to a pandas DataFrame
faq_df = pd.DataFrame(list(faq_data.items()), columns=['Question', 'Answer'])

# Basic text preprocessing
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

faq_df['Question'] = faq_df['Question'].apply(preprocess_text)

# Convert text data to numerical data using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(faq_df['Question']).toarray()

def get_response(user_input):
    user_input_processed = preprocess_text(user_input)
    user_input_vec = vectorizer.transform([user_input_processed]).toarray()
    
    # Compute cosine similarity between user input and FAQ questions
    similarities = cosine_similarity(user_input_vec, X)
    closest_match_idx = similarities.argmax()
    
    # If the similarity is low, we assume the bot doesn't know the answer
    if similarities[0][closest_match_idx] < 0.2:
        return "Sorry, I do not understand your question."
    else:
        return faq_df.iloc[closest_match_idx]['Answer']

print("Welcome to the voice-enabled FAQ chatbot!")
speak("Welcome to the voice-enabled FAQ chatbot!")

while True:
    user_input = recognize_speech()
    if user_input:
        if user_input.lower() == 'exit':
            speak("Goodbye!")
            break
        response = get_response(user_input)
        print(f"Bot: {response}")
        speak(response)
