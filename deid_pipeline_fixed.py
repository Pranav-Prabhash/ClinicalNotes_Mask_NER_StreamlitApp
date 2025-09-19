import re
import spacy

# Load spaCy small English model
nlp = spacy.load("en_core_web_sm")

# List of common Indian cities for regex masking
indian_cities = [
    "Chennai", "Mumbai", "Delhi", "Bangalore", "Kolkata",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"
]
cities_pattern = r"\b(" + "|".join(indian_cities) + r")\b"

# Whitelist of clinical keywords that must NOT be masked
CLINICAL_KEYWORDS = {"fever", "cough", "diabetes", "cancer", "headache", "asthma"}

def deidentify_text(text: str) -> str:
    """
    De-identify sensitive information from clinical/patient text.
    """

    new_text = text

    # --- 1️⃣ Regex patterns ---
    numeric_dates = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"
    month_dates = (
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},?\s\d{4}\b"
        r"|\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s\d{4}\b"
    )
    phone_pattern = r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4}"
    email_pattern = r"\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b"
    id_pattern = r"\b(?:[A-Z]{2,10}-\d{3,10}|\d{3,10}-[A-Z]{1,10}|[A-Z]{2}\d{5,10}|\d{5,10}[A-Z]{2})\b"

    address_pattern = (
        r"\b\d{1,5}\s[\w\s]+"
        r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Boulevard|Blvd|Drive|Dr|Court|Ct|MG Road|Main Road)?"
        r"[,.\s]*[A-Za-z\s]*"
        r"(?:\d{3,6})?\b"
    )

    # --- 2️⃣ Apply regex replacements ---
    new_text = re.sub(numeric_dates, "[DATE]", new_text)
    new_text = re.sub(month_dates, "[DATE]", new_text, flags=re.IGNORECASE)
    new_text = re.sub(phone_pattern, "[CONTACT]", new_text)
    new_text = re.sub(email_pattern, "[CONTACT]", new_text)
    new_text = re.sub(id_pattern, "[ID]", new_text)
    new_text = re.sub(address_pattern, "[ADDRESS]", new_text, flags=re.IGNORECASE)
    new_text = re.sub(cities_pattern, "[ADDRESS]", new_text, flags=re.IGNORECASE)

    # --- 3️⃣ Apply SpaCy NER replacements ---
    doc = nlp(new_text)
    for ent in doc.ents:
        # Skip already masked
        if ent.text.startswith("[") and ent.text.endswith("]"):
            continue

        # Skip masking clinical keywords
        if ent.text.lower() in CLINICAL_KEYWORDS:
            continue

        if ent.label_ == "PERSON":
            new_text = new_text.replace(ent.text, "[NAME]")
        elif ent.label_ == "ORG":
            if any(word in ent.text.lower() for word in ["hospital", "clinic", "insurance"]):
                new_text = new_text.replace(ent.text, "[ORG]")
        elif ent.label_ in ["GPE", "LOC", "FAC"]:
            new_text = new_text.replace(ent.text, "[ADDRESS]")
        elif ent.label_ == "DATE":
            new_text = new_text.replace(ent.text, "[DATE]")

    return new_text
