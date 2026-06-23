import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

questions = [
    "What is AI?",
    "What is Machine Learning?",
    "What is Python?",
    "What is NLP?"
]

answers = [
    "AI stands for Artificial Intelligence.",
    "Machine Learning is a subset of AI.",
    "Python is a programming language.",
    "NLP stands for Natural Language Processing."
]

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join(tokens)

processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

print("FAQ Chatbot Started!")
print("Type 'exit' to quit.")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    processed_input = preprocess(user_input)

    user_vector = vectorizer.transform([processed_input])
    similarity = cosine_similarity(user_vector, faq_vectors)

    index = similarity.argmax()

    print("Bot:", answers[index])
