# config/signals.py
from typing import TypedDict, Dict


class SignalConfig(TypedDict):
    name: str
    description: str
    prompt: str


CREDIBILITY_SIGNALS: Dict[str, SignalConfig] = {
    "evidence": {
        "name": "evidence",
        "description": "Does this text fail to present any supporting evidence or arguments to substantiate its claims?",
        "prompt": "Does this text fail to present any supporting evidence or arguments to substantiate its claims?",
    },
    "bias": {
        "name": "bias",
        "description": "Does this text contain explicit or implicit biases, such as confirmation bias, selection bias, or framing bias?",
        "prompt": "Does this text contain explicit or implicit biases, such as confirmation bias, selection bias, or framing bias?",
    },
    "inference": {
        "name": "inference",
        "description": "Does this text make claims about correlation and causation?",
        "prompt": "Does this text make claims about correlation and causation?",
    },
    "polarising_language": {
        "name": "polarising_language",
        "description": "Does this text use polarising terms or create divisions into sharply contrasting groups, opinions, or beliefs?",
        "prompt": "Does this text use polarising terms or create divisions into sharply contrasting groups, opinions, or beliefs?",
    },
    "document_citation": {
        "name": "document_citation",
        "description": "Does this text lack citations of studies or documents to support its claims?",
        "prompt": "Does this text lack citations of studies or documents to support its claims?",
    },
    "informal_tone": {
        "name": "informal_tone",
        "description": "Does this text use informal tone elements like all caps, consecutive exclamation marks, or question marks?",
        "prompt": "Does this text use informal tone elements like all caps, consecutive exclamation marks, or question marks?",
    },
    "explicitly_unverified_claims": {
        "name": "explicitly_unverified_claims",
        "description": "Does this text contain claims that explicitly lack confirmation?",
        "prompt": "Does this text contain claims that explicitly lack confirmation?",
    },
    "personal_perspective": {
        "name": "personal_perspective",
        "description": "Does this text include the author’s personal opinions rather than factual reporting?",
        "prompt": "Does this text include the author’s personal opinions rather than factual reporting?",
    },
    "emotional_valence": {
        "name": "emotional_valence",
        "description": "Does the language of this text carry strong emotional valence, either predominantly negative or positive, rather than being neutral?",
        "prompt": "Does the language of this text carry strong emotional valence, either predominantly negative or positive, rather than being neutral?",
    },
    "call_to_action": {
        "name": "call_to_action",
        "description": "Does this text contain language that can be interpreted as a call to action, telling readers what to do or follow through with a specific task?",
        "prompt": "Does this text contain language that can be interpreted as a call to action, telling readers what to do or follow through with a specific task?",
    },
    "expert_citation": {
        "name": "expert_citation",
        "description": "Does this text lack citations of experts in the subject?",
        "prompt": "Does this text lack citations of experts in the subject?",
    },
    "clickbait": {
        "name": "clickbait",
        "description": "Does this text's title contain sensationalised or misleading headlines to attract clicks?",
        "prompt": "Does this text's title contain sensationalised or misleading headlines to attract clicks?",
    },
    "incorrect_spelling": {
        "name": "incorrect_spelling",
        "description": "Does this text contain significant misspellings and/or grammatical errors?",
        "prompt": "Does this text contain significant misspellings and/or grammatical errors?",
    },
    "misleading_about_content": {
        "name": "misleading_about_content",
        "description": "Does this text's title emphasise different information than the body topic?",
        "prompt": "Does this text's title emphasise different information than the body topic?",
    },
    "incivility": {
        "name": "incivility",
        "description": "Does this text use stereotypes and/or generalisations of groups of people?",
        "prompt": "Does this text use stereotypes and/or generalisations of groups of people?",
    },
    "impoliteness": {
        "name": "impoliteness",
        "description": "Does this text contain insults, name-calling, or profanity?",
        "prompt": "Does this text contain insults, name-calling, or profanity?",
    },
    "sensationalism": {
        "name": "sensationalism",
        "description": "Does this text present information in a manner designed to evoke strong emotional reactions?",
        "prompt": "Does this text present information in a manner designed to evoke strong emotional reactions?",
    },
    "source_credibility": {
        "name": "source_credibility",
        "description": "Does this text cite low-credibility sources?",
        "prompt": "Does this text cite low-credibility sources?",
    },
    "reported_by_other_sources": {
        "name": "reported_by_other_sources",
        "description": "Does this text present a story that was not reported by other reputable media outlets?",
        "prompt": "Does this text present a story that was not reported by other reputable media outlets?",
    },
}
