import re
import sys
import joblib

MODEL_PATH = "genre_classifier_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_genre(description, model, vectorizer):
    cleaned = clean_text(description)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        top3_idx = proba.argsort()[-3:][::-1]
        top3 = [(model.classes_[i], round(proba[i] * 100, 1)) for i in top3_idx]
        return prediction, top3
    return prediction, None


def main():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    if "--interactive" in sys.argv:
        print("Type a plot summary and press Enter (or 'quit' to exit):\n")
        while True:
            text = input("> ")
            if text.strip().lower() in ("quit", "exit"):
                break
            genre, top3 = predict_genre(text, model, vectorizer)
            print(f"Predicted genre: {genre}")
            if top3:
                print(f"Top 3: {top3}")
            print()
        return

    demo_examples = [
        " A brilliant but cynical detective is tasked with investigating a string of bizarre, ritualistic murders in a decaying city. As he delves deeper into the grotesque crime scenes, he uncovers a horrifying conspiracy involving a serial killer who uses the seven deadly sins as his twisted modus operandi.",
        "After a minor fender bender, two strangers engage in a petty, escalating road rage incident that completely spirals out of control. Their obsessive, mutual vendetta rapidly destroys their personal lives, careers, and sanity as they stop at nothing to get revenge.",
        "In a dystopian future where humanity is sterile, a disillusioned bureaucrat unexpectedly becomes the protector of the world's first pregnant woman in nearly two decades. Together, they must evade relentless government forces and violent rebel factions to reach a rumored sanctuary at sea."
    ]

    print(f"{'Plot summary':<90} | Predicted Genre")
    print("-" * 115)
    for text in demo_examples:
        genre, _ = predict_genre(text, model, vectorizer)
        display_text = (text[:85] + "...") if len(text) > 88 else text
        print(f"{display_text:<90} | {genre}")


if __name__ == "__main__":
    main()
