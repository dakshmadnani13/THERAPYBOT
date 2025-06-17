import speech_recognition as sr
import pyttsx3 as py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
reply_data = [
    ("I feel really low today [SEP] sad", "I'm here for you. Want to talk more about what happened?"),
    ("I'm so pumped about my trip [SEP] happy", "That’s amazing! Where are you headed?"),
    ("I feel overwhelmed and shaky [SEP] anxious", "Let’s take a deep breath together. You're safe."),
    ("I'm okay I guess [SEP] neutral", "Thanks for checking in. I'm here if you ever want to chat.")
]
# ✅ Training data
training_sentences = [
    "I feel really low today",
    "I'm just so sad and tired",
    "Nothing feels right anymore",
    "I'm excited for the weekend!",
    "Today was amazing",
    "I'm so pumped about this trip",
    "I'm having a panic attack",
    "I feel anxious and overwhelmed",
    "I’m really nervous about tomorrow",
    "I'm fine I guess",
    "I’m not sure how I feel right now"
]

training_labels = [
    "sad", "sad", "sad",
    "happy", "happy", "happy",
    "anxious", "anxious", "anxious",
    "neutral", "neutral"
]

# ✅ Convert sentences to vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_sentences)

# ✅ Train the classifier
clf = LogisticRegression()
clf.fit(X, training_labels)

# ✅ Start the speech engine
engine = py.init()
recognizer = sr.Recognizer()

# ✅ Begin voice input
with sr.Microphone() as source:
    print("🎤 Listening...")
    audio = recognizer.listen(source)

    try:
        # 🔁 Speech to Text
        text = recognizer.recognize_google(audio)
        print("🧠 You said:", text)

        # 🔁 Text to Vector
        vector = vectorizer.transform([text])

        # 🔁 Predict Emotion
        prediction = clf.predict(vector)
        emotion = prediction[0]
        print("🔍 Predicted Emotion:", emotion)

        # 🔁 Choose reply
        if emotion == "sad":
            reply = "I'm really sorry you're feeling this way. Want to talk more about it?"
        elif emotion == "happy":
            reply = "That's amazing! Tell me what made you happy!"
        elif emotion == "anxious":
            reply = "It’s okay to feel anxious. Let's take a deep breath together."
        else:
            reply = "I'm here for you. Thanks for sharing that."

        # 🗣️ Speak reply
        engine.say(reply)
        engine.runAndWait()

    except sr.UnknownValueError:
        engine.say("Sorry, I couldn't understand what you said.")
        engine.runAndWait()
    except sr.RequestError:
        engine.say("There was a problem connecting to the speech service.")
        engine.runAndWait()
  