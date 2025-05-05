# config/signals.py
from typing import TypedDict, Dict


class SignalConfig(TypedDict):
    name: str
    description: str
    question: str


CREDIBILITY_SIGNALS: Dict[str, SignalConfig] = {
    "evidence": {
        "name": "evidence",
        "question": "Does this text fail to present any supporting evidence or arguments to substantiate its claims?",
    },
    "bias": {
        "name": "bias",
        "question": "Does this text contain explicit or implicit biases, such as confirmation bias, selection bias, or framing bias?",
    },
    "inference": {
        "name": "inference",
        "question": "Does this text make claims about correlation and causation?",
    },
    "polarising_language": {
        "name": "polarising_language",
        "question": "Does this text use polarising terms or create divisions into sharply contrasting groups, opinions, or beliefs?",
    },
    "document_citation": {
        "name": "document_citation",
        "question": "Does this text lack citations of studies or documents to support its claims?",
    },
    "informal_tone": {
        "name": "informal_tone",
        "question": "Does this text use informal tone elements like all caps, consecutive exclamation marks, or question marks?",
    },
    "explicitly_unverified_claims": {
        "name": "explicitly_unverified_claims",
        "question": "Does this text contain claims that explicitly lack confirmation?",
    },
    "personal_perspective": {
        "name": "personal_perspective",
        "question": "Does this text include the author's personal opinions rather than factual reporting?",
    },
    "emotional_valence": {
        "name": "emotional_valence",
        "question": "Does the language of this text carry strong emotional valence, either predominantly negative or positive, rather than being neutral?",
    },
    "call_to_action": {
        "name": "call_to_action",
        "question": "Does this text contain language that can be interpreted as a call to action, telling readers what to do or follow through with a specific task?",
    },
    "expert_citation": {
        "name": "expert_citation",
        "question": "Does this text lack citations of experts in the subject?",
    },
    "clickbait": {
        "name": "clickbait",
        "question": "Does this text's title contain sensationalised or misleading headlines to attract clicks?",
    },
    "incorrect_spelling": {
        "name": "incorrect_spelling",
        "question": "Does this text contain significant misspellings and/or grammatical errors?",
    },
    "misleading_about_content": {
        "name": "misleading_about_content",
        "question": "Does this text's title emphasise different information than the body topic?",
    },
    "incivility": {
        "name": "incivility",
        "question": "Does this text use stereotypes and/or generalisations of groups of people?",
    },
    "impoliteness": {
        "name": "impoliteness",
        "question": "Does this text contain insults, name-calling, or profanity?",
    },
    "sensationalism": {
        "name": "sensationalism",
        "question": "Does this text present information in a manner designed to evoke strong emotional reactions?",
    },
    "source_credibility": {
        "name": "source_credibility",
        "question": "Does this text cite low-credibility sources?",
    },
    "reported_by_other_sources": {
        "name": "reported_by_other_sources",
        "question": "Does this text present a story that was not reported by other reputable media outlets?",
    },
}

CREDIBILITY_SIGNALS_CONDENSED = {
    # ——— Factual-evidence signals ———
    "evidence_present": {
        "name": "evidence_present",
        "signal_type": "factual_evidence",
        "question": (
            "Does the text quote or link at least one **named** source, data table, "
            "document or eyewitness statement that supports its *main* claim?"
        ),
        "polarity": "POSITIVE",
    },
    "explicit_unverified_claims": {
        "name": "explicit_unverified_claims",
        "signal_type": "factual_evidence",
        "question": (
            "Does the text include a factual statement that the author explicitly says "
            "is **unverified / unconfirmed** (e.g. 'police have not confirmed')?"
        ),
        "polarity": "NEGATIVE",
    },
    "inference_error": {
        "name": "inference_error",
        "signal_type": "factual_evidence",
        "question": (
            "Does the text state a causal or quantitative conclusion **without any supporting "
            "evidence** provided in the article?"
        ),
        "polarity": "NEGATIVE",
    },
    # ——— Source-quality signals ———
    "credible_sourcing": {
        "name": "credible_sourcing",
        "signal_type": "source_quality",
        "question": (
            "Does the article rely **primarily** on sources with recognised expertise or authority "
            "on the topic (named officials, peer-reviewed papers, major institutions)?"
        ),
        "polarity": "POSITIVE",
    },
    "external_corroboration": {
        "name": "external_corroboration",
        "signal_type": "source_quality",
        "question": (
            "Is the same core fact reported by **at least one other** independent, reputable outlet "
            "or official record that is cited or linked in the text?"
        ),
        "polarity": "POSITIVE",
    },
    # ——— Style / tone risk signals ———
    "strong_framing_tone": {
        "name": "strong_framing_tone",
        "signal_type": "style_tone",
        "question": (
            "Does the text use emotion-laden or vilifying words aimed at persuading rather than informing?"
        ),
        "polarity": "WEAK_NEGATIVE",
    },
    "clickbait": {
        "name": "clickbait",
        "signal_type": "style_tone",
        "question": (
            "Does the **headline** employ exaggeration, curiosity-gaps, all-caps or shock phrases to lure clicks?"
        ),
        "polarity": "WEAK_NEGATIVE",
    },
    "writing_quality_alert": {
        "name": "writing_quality_alert",
        "signal_type": "style_tone",
        "question": (
            "Does the body contain **multiple** spelling errors, all-caps yelling, profanity or texting abbreviations?"
        ),
        "polarity": "WEAK_NEGATIVE",
    },
    "misleading_about_content": {
        "name": "misleading_about_content",
        "signal_type": "style_tone",
        "question": (
            "After reading the body, does the headline focus on a **different topic or outcome** "
            "than the article actually describes?"
        ),
        "polarity": "WEAK_NEGATIVE",
    },
    
}
