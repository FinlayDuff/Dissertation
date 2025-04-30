misinformation_dict = {True: "Fake", False: "Credible"}
veracity_dict = {
    "true": 1,
    "mostly-true": 1,
    "half-true": 1,
    "mostly-false": 0,
    "false": 0,
    "pants-fire": 0,
}
fake_real_label_dict = {"Fake": 0, "Real": 1}


POLARITY = {"POSITIVE": 1.0, "NEGATIVE": -1.0, "WEAK_NEGATIVE": -0.5}
CONFIDENCE = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}
RELEVANCE = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.25}
TRUST = {"HIGH": 1.0, "MEDIUM": 0.75, "LOW": 0.5}
LABEL = {"TRUE": 1.0, "FALSE": -1.0}
