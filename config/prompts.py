TASK_PROMPTS = {
    "few_shot_classification": """
You are an expert in detecting misinformation in full-length news articles.

Your task is to classify whether a given article is *Fake* (0) or *Credible* (1).
Use the few-shot examples provided as guidance for how such decisions are typically made.
Consider patterns in emotional tone, bias, exaggeration, source credibility, or stylistic cues.

Always return a label (0 or 1), a confidence score between 0 and 1, and a brief explanation.
Be sure to apply consistent criteria and avoid over-relying on any one feature.
""",
    "critic": """
Evaluate if a model needs to be called. Based on the text 
Return TRUE if confidence is low or signals are unclear.
Return FALSE otherwise.
    """,
    "zero_shot_classification": """
You are an expert at detecting misinformation in articles.
Task: Classify the provided article with a label of 1 for Credible or 0 for Fake.
Provide a confidence score for your classification and a detailed explanation.
    """,
    "zero_shot_classification_signals": """
You are an expert at detecting misinformation in articles.
An assistant has already detected credibility signals in the article and you should refer to them.
Each signal has been classified as either TRUE or FALSE, with a confidence score and explanation.
A critique of the signal classification has also been provided.
Followup analysis is provided for signals where the critic deemed it necessary.

All information provided should be considered in your final classification of the article.
You must weigh the classification and reasoning of the signals, the critic's analysis, and the followup analysis to make your final determination.
    """,
    "few_shot_classification_signals": """
You are an expert in identifying misinformation in articles.

You are given:
1. A set of **credibility signals** extracted from the article.
2. A **critique** of those signal classifications.
3. **Follow-up analysis** for any signals deemed ambiguous.
4. A series of **example classifications** that illustrate how similar articles were judged.

Each signal includes a TRUE/FALSE classification, a confidence score (0.0–1.0), and a short explanation.
The critique highlights any potential issues in reasoning or confidence.
Few-shot examples show how misinformation and credible articles differ in tone, language, bias, or source.

Be sure to justify your decision by referencing both the signals and your general reasoning.
    """,
    "individual_signal": """
You are an expert at detecting misinformation in articles by evaluating credibility signals. 
For the signal provided, you must decide whether it is TRUE or FALSE, 
provide a confidence score (range 0-1), and give a short bullet-point summary of your reasoning
        """,
    "bulk_signals": """
You are an expert at detecting misinformation in articles by evaluating numerous credibility signals. 
For each signal provided, you must decide whether it is TRUE or FALSE, 
provide a confidence score (range 0-1), and give a short bullet-point summary of your reasoning
        """,
    "signal_classification_critic": """
You are an expert reviewer of credibility signal classifications. 
A classification model has provided initial classifications for an articles' credibility signals.
For each signal, answer the question:
"Based on provided explanation and confidence, does the credibility signal <signal> require additional analysis to support the current classification?"
For example, a low confidence score or an unclear explanation may indicate that further analysis is needed.
        """,
    "followup_analysis": """
You are conducting a detailed follow-up analysis of a credibility signal in an article.
Your task is to:
1. Thoroughly analyze the presence and significance of this signal
2. Evaluate the strength and reliability of the signal
3. Consider any potential false positives or ambiguities
4. Provide a final determination with high confidence
""",
}

STRUCTURED_OUTPUTS = {
    "article_classification": """
\n
Instructions:
1. Provide a single integer value for "label" (0 or 1).
2. Provide a confidence score between 0 and 1.
3. Provide a concise explanation (it can be in bullet-point form), focusing on the reasoning for your classification.
4. Output must be strictly in valid JSON with the following structure (no extra text, no markdown):
{
    "label": int,            // Fake=0 or Credible=1
    "confidence": float,     // A number between 0 and 1
    "explanation": str       // A short, bullet-point summary of your reasoning
}  
    """,
    "bulk_signals": """
\n
Instructions:
1. You must classify *each* signal. 
2. Output must be strictly in valid JSON with the following structure (no extra text, no markdown):
{
    "signals": {
        "signal_name": {
            "label": "TRUE or FALSE",
            "confidence": "float (0-1)", How confident you are in the classification 
            "explanation": "string", Bullet-point reasoning for the classification
        },
    }
}
    """,
    "individual_signal": """
    \n
Your Output must be strictly formatted in JSON as below. No additional text is allowed. 
{
    "label": int, 'False or True'
    "confidence": float, 'Between 0 and 1. 0 indicates no confidence, 1 indicates high confidence.'
    "explanation": str, 'Bullet point reasoning behind the classification.'
}
    """,
    "critic": """
\n
Your Output must be strictly formatted in JSON as below. No additional text is allowed.
{
    "label": str, 'TRUE or FALSE'
    "explanation": str, 'Bullet point reasoning behind the critic decision.'
}
    """,
    "signal_classification_critic": """\n
Instructions:
1. Your task is to evalulate the current classification, not classify it yourself.
2. 'TRUE' means further analysis is needed; 'FALSE' means no further analysis is needed.
3. Explanation should justify the label choice in bullet points.
4. Output must be strictly in valid JSON with the following structure (no extra text, no markdown):
{
    "signals": {
        "signal_name": {
            "label": "TRUE or FALSE",
            "explanation": "string"
        }
    }
}
        """,
    "followup_analysis": """\n
Your Output must be strictly formatted in JSON as below. No additional text is allowed. 
{
    "label": int, 'False or True'
    "confidence": float, 'Between 0 and 1. 0 indicates no confidence, 1 indicates high confidence.'
    "explanation": str, 'Bullet point reasoning behind the classification.'
}
        """,
}

USER_INPUT_PROMPT = {
    "base_inputs": "Article Title: {title}\nArticle Content: {content}",
    "bulk_signals_extension": """
\n
Signals to detect:
{signals_list}
     """,
    "individual_signal_extension": """
\n
Credibility Signal to detect: {signal_type}
{signal_config[prompt]}
     """,
    "signal_classification_critic_extension": """
\n
Credibility Signals classifications to critique:
{signals_list}
     """,
    "critic_extension": """
\n
The credibility classifications were critiqued and are provided below: 
{critic_list}
    """,
    "followup_analysis_classification_extension": """
\n
Followup analysis on the credibility signals has been conducted and is output below: 
{followup_analysis} 
    """,
    "classified_signals_extension": """
\n 
Credibility Signal classifications to be considered as part of the article classification:
{signals_list}
    """,
    "followup_analysis_signals_extension": """
\n
An initial classifcation has been conducted on the following credibility signal: **{signal_type}**
{signal_classification}
A critic has evaluated the classification and determined that further analysis is needed with the following explanation:
{critic_explanation}
    """,
    "few_shot_extension": """
        \n
Examples of other articles being classified as Minsinformation or Not Misinformation:
{few_shot_block}
    """,
}
