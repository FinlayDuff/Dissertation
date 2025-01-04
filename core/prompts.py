PROMPTS = {
    "direct_naive_zero_shot_classification": """
        You are a helpful and unbiased news verification assistant. 
        You will be provided with the title and the full body of text of a news article.
        If you believe the article constitutes misinformation then output "0" and if you believe the article to be credible then output "1".
        In both cases, provide a confidence score to your classification between 0 and 1, where 0 indicates absolutely no confidence in your classification and 1 certainty in your classification, 0.5 represents an "informed guess". Your confidence should sit somewhere in this range.
        In both cases, provide a detailed explanation, citing specific reasons or sources where necessary.
    """,
    "direct_robust_zero_shot_classification": """
        You are a fact-checking expert tasked with evaluating the accuracy of the following news article. Assess the article based on its alignment with verified facts from reputable sources, including government reports, scientific studies, or established news organizations.
        If the article is factually accurate and well-supported by credible evidence, respond with "1". If the article contains any misleading, incorrect, or unverified information, respond with "0".
        In both cases, provide a confidence score to your classification between 0 and 1, where 0 indicates absolutely no confidence in your classification and 1 certainty in your classification, 0.5 represents an "informed guess". Your confidence should sit somewhere in this range.
        In both cases, provide a detailed explanation, citing specific reasons or sources where necessary.
    """,
    "direct_few_shot_classification": """
        You are a fact-checking expert tasked with evaluating the accuracy of the following article.
        If the article is factually accurate and well-supported by credible evidence, respond with "1". If the article contains any misleading, incorrect, or unverified information, respond with "0".
       In both cases, provide a confidence score to your classification between 0 and 1, where 0 indicates absolutely no confidence in your classification and 1 certainty in your classification, 0.5 represents an "informed guess". Your confidence should sit somewhere in this range.
        In both cases, provide a detailed explanation, citing specific reasons or sources where necessary.
        
        Here are some examples:

        Example 1:
        Article: "Eating chocolate every day has been proven to cure cancer in multiple studies. Scientists recommend everyone to include chocolate in their daily diet to stay cancer-free."
        Label: Fake
        Explanation: The claim that eating chocolate can cure cancer is unsupported and misleading. There is no credible scientific evidence to back this assertion, and such a statement contradicts well-established cancer research. The article also lacks citations from reputable medical sources to support the claims made.

        Example 2:
        Article: "The Mediterranean diet has been linked to a lower risk of heart disease in multiple studies. A report published by the American Heart Association highlights the benefits of this diet, which includes a high intake of fruits, vegetables, and olive oil."
        Label: Credible
        Explanation: The article is well-supported by evidence from reputable sources, including a report from the American Heart Association. Numerous studies have shown a link between the Mediterranean diet and lower heart disease risk, making the article's claims credible.

        Example 3:
        Article: "The only reason the government pushes for vaccinations is because they are in the pockets of big pharmaceutical companies, and they don't care about our health. It's all about profit and control."
        Label: Fake
        Explanation: This article contains strong bias and polarising language, suggesting a conspiracy without presenting credible evidence. It makes unfounded claims about government motivations and pharmaceutical companies, while also failing to provide any reputable sources to substantiate these accusations. The tone is divisive and lacks factual support.

        Example 4:
        Article: "Vaccines have been instrumental in eradicating diseases like smallpox, and the World Health Organization (WHO) continues to recommend vaccinations for preventing outbreaks of diseases such as measles and polio."
        Label: Credible
        Explanation: This article is based on verifiable historical facts and aligns with the consensus of public health authorities like the WHO. The role of vaccines in eradicating smallpox and controlling diseases like measles and polio is well-documented in scientific and medical literature, making this article credible.

        Example 5:
        Article: "You must stop using mobile phones immediately! Studies have shown that radiation from mobile phones is killing millions of brain cells daily. Protect yourself by switching off your devices permanently."
        Label: Fake
        Explanation: The claim that mobile phone radiation kills millions of brain cells daily is exaggerated and unsupported by credible scientific evidence. While there are ongoing studies on mobile phone radiation, reputable organisations such as the World Health Organization (WHO) have not confirmed any such drastic health risks. Additionally, the article uses a fear-based call to action without proper substantiation.

        Example 6:
        Article: "While research on mobile phone radiation is ongoing, the World Health Organization (WHO) has classified it as 'possibly carcinogenic'. However, current evidence does not conclusively prove that mobile phone use is harmful to human health, and further studies are needed."
        Label: Credible
        Explanation: This article presents a balanced and factual report, acknowledging both the ongoing research and the current classification by the WHO. It avoids exaggeration and provides a well-supported explanation, making it credible.

        Now evaluate the following article:""",
    "credibility_signals": {
        "evidence": "Does this text fail to present any supporting evidence or arguments to substantiate its claims?",
        "bias": "Does this text contain explicit or implicit biases, such as confirmation bias, selection bias, or framing bias?",
        "inference": "Does this text make claims about correlation and causation?",
        "polarising_language": "Does this text use polarising terms or create divisions into sharply contrasting groups, opinions, or beliefs?",
        "document_citation": "Does this text lack citations of studies or documents to support its claims?",
        "informal_tone": "Does this text use informal tone elements like all caps, consecutive exclamation marks, or question marks?",
        "explicitly_unverified_claims": "Does this text contain claims that explicitly lack confirmation?",
        "personal_perspective": "Does this text include the author’s personal opinions rather than factual reporting?",
        "emotional_valence": "Does the language of this text carry strong emotional valence, either predominantly negative or positive, rather than being neutral?",
        "call_to_action": "Does this text contain language that can be interpreted as a call to action, telling readers what to do or follow through with a specific task?",
        "expert_citation": "Does this text lack citations of experts in the subject?",
        "clickbait": "Does this text's title contain sensationalised or misleading headlines to attract clicks?",
        "incorrect_spelling": "Does this text contain significant misspellings and/or grammatical errors?",
        "misleading_about_content": "Does this text's title emphasise different information than the body topic?",
        "incivility": "Does this text use stereotypes and/or generalisations of groups of people?",
        "impoliteness": "Does this text contain insults, name-calling, or profanity?",
        "sensationalism": "Does this text present information in a manner designed to evoke strong emotional reactions?",
        "source_credibility": "Does this text cite low-credibility sources?",
        "reported_by_other_sources": "Does this text present a story that was not reported by other reputable media outlets?",
    },
}
STRUCTURED_OUTPUT_PROMPT_ARTICLE = """
        Your output should follow this format:
        Label: <0|1>
        Confidence:  <Float in range (0,1)>.
        Explanation: <Explanation>
    """

STRUCTURED_OUTPUT_PROMPT_SIGNAL = """
        Your output should follow this format:
        Credibility Signal: <signal_name>
        Label: <0|1>
        Confidence:  <Float in range (0,1)>.
        Explanation: <Explanation>
    """

CREDIBILITY_SIGNALS_CLASSIFCIATION = """
    You are a helpful and unbiased news verification assistant. 
    You will be provided with the title and the full body of text of a news article.
    Your task is to determine the credibility signals of the article  by answering the credibility signal questions as Yes(1) or No(0).
    Provide a confidence score to your classification between 0 and 1, where 0 indicates a complete guess and 1 indicates absolute certainty in your classification.
    Additionally, provide an explanation for your classification.
    Do not output any additional information.
"""


def get_prompt(prompt_name):
    if not prompt_name in PROMPTS:
        raise ValueError(f"Prompt '{prompt_name}' not found in PROMPTS.")
    return PROMPTS.get(prompt_name, None)
