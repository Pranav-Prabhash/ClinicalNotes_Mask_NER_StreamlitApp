# entity_pipeline.py

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

# Load spaCy small English model
nlp = spacy.load("en_core_web_sm")

# ------------------------------
# Custom lists of entities
# ------------------------------

# Diseases
diseases = [
    "Diabetes", "Hypertension", "Asthma", "COVID-19", "Pneumonia",
    "Tuberculosis", "Malaria", "Cancer", "Stroke", "Heart Disease",
    "Kidney Failure", "HIV", "Hepatitis B", "Hepatitis C",
    "Arthritis", "Depression", "Anxiety", "Migraine",
    "Thyroid Disorder", "Obesity", "Epilepsy", "Alzheimer's"
]

# Medications
medications = [
    "Metformin", "Paracetamol", "Insulin", "Aspirin", "Ibuprofen",
    "Amoxicillin", "Azithromycin", "Atorvastatin", "Omeprazole",
    "Losartan", "Hydrochlorothiazide", "Prednisone", "Warfarin",
    "Metoprolol", "Salbutamol", "Diclofenac", "Levothyroxine",
    "Ciprofloxacin", "Doxycycline", "Tramadol"
]

# Symptoms
symptoms = [
    "Fever", "Cough", "Headache", "Fatigue", "Chest pain",
    "Shortness of breath", "Nausea", "Vomiting", "Dizziness",
    "Sore throat", "Runny nose", "Diarrhea", "Back pain",
    "Joint pain", "Swelling", "Rash", "Anxiety",
    "Palpitations", "Weight loss", "Loss of appetite"
]

# Lab Tests / Results
lab_tests = [
    "Blood Glucose", "WBC Count", "Hemoglobin", "Creatinine",
    "Platelet Count", "ECG", "MRI", "CT scan", "X-ray",
    "Liver Function Test", "Kidney Function Test", "Urinalysis",
    "HbA1c", "Cholesterol", "Triglycerides", "Thyroid Function Test",
    "Blood Pressure", "Oxygen Saturation", "Echocardiogram",
    "Prothrombin Time"
]

# Procedures
procedures = [
    "MRI", "Appendectomy", "Chemotherapy", "Vaccination", "CT scan",
    "Dialysis", "Angioplasty", "Bypass Surgery", "Endoscopy",
    "Biopsy", "Cesarean Section", "Cataract Surgery", "Liver Transplant",
    "Knee Replacement", "Hip Replacement", "Colonoscopy", "Radiotherapy",
    "Pacemaker Implantation", "Tonsillectomy", "Gastrectomy"
]

# ------------------------------
# PhraseMatcher setup
# ------------------------------
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")  # case-insensitive matching

def add_list_to_matcher(label, terms):
    patterns = [nlp.make_doc(t) for t in terms]
    matcher.add(label, patterns)

add_list_to_matcher("DISEASE", diseases)
add_list_to_matcher("MEDICATION", medications)
add_list_to_matcher("SYMPTOM", symptoms)
add_list_to_matcher("LAB_TEST", lab_tests)
add_list_to_matcher("PROCEDURE", procedures)

# ------------------------------
# Main function: extract entities
# ------------------------------
def extract_entities_from_text(text: str) -> pd.DataFrame:
    """
    Extract entities using PhraseMatcher lists.
    Returns a DataFrame with columns: ['Entity','Type','Start','End'].
    """
    doc = nlp(text)
    entities = []

    # PhraseMatcher matches
    matches = matcher(doc)
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span = doc[start:end]
        entities.append((span.text, label, span.start_char, span.end_char))

    # Remove duplicates by (start,end,label)
    seen = set()
    filtered = []
    for ent_text, ent_label, s, e in entities:
        key = (s, e, ent_label)
        if key not in seen:
            seen.add(key)
            filtered.append((ent_text, ent_label, s, e))

    df = pd.DataFrame(filtered, columns=["Entity", "Type", "Start", "End"])
    return df


