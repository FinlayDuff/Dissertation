STRUCTURED_OUTPUT_PROMPT = """
    Your output should follow this format:
    Label: <Credible|Fake>
    Explanation: Provide a structured explanation, specifying which parts of the article are accurate or misleading and why, along with any relevant sources or reasoning.
"""

NAIVE_ZERO_SHOT_CLASSIFICATION_PROMPT = """
    You are a fact-checking expert tasked with evaluating the accuracy of the following article.
    If the article is factually accurate and well-supported by credible evidence, respond with "Credible". If the article contains any misleading, incorrect, or unverified information, respond with "Fake". In both cases, provide a detailed explanation, citing specific reasons or sources where necessary.
"""

ROBUST_ZERO_SHOT_CLASSIFICATION_PROMPT = """
    You are a fact-checking expert tasked with evaluating the accuracy of the following news article. Assess the article based on its alignment with verified facts from reputable sources, including government reports, scientific studies, or established news organizations.
    If the article is factually accurate and well-supported by credible evidence, respond with "Credible". If the article contains any misleading, incorrect, or unverified information, respond with "Fake". In both cases, provide a detailed explanation, citing specific reasons or sources where necessary.
"""
