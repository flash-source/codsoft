import re
import string
import joblib

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/digits, drop stopwords. Must match the
    cleaning used in spam_sms_detection.ipynb, or the vectorizer's learned
    vocabulary won't line up with the input."""
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', ' ', text)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return ' '.join(tokens)


def load_model():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('spam_classifier_model.pkl')
    return vectorizer, model


def predict(message: str, vectorizer, model) -> str:
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return 'spam' if pred == 1 else 'ham'


if __name__ == '__main__':
    vectorizer, model = load_model()

    sample_messages = [
        "Congratulations! You've won a free cruise. Call now to claim your prize!",
        "Hey, are we still on for dinner tonight?",
        "URGENT: Your account has been suspended. Verify now at the link below.",
        "Can you pick up milk on your way home?",
    ]

    print("=== Sample predictions ===")
    for msg in sample_messages:
        print(f"[{predict(msg, vectorizer, model):>4}] {msg}")

    print("\n=== Try your own message (type 'quit' to exit) ===")
    while True:
        user_input = input("\nEnter an SMS message: ").strip()
        if user_input.lower() in ('quit', 'exit', ''):
            break
        print(f"Prediction: {predict(user_input, vectorizer, model)}")
