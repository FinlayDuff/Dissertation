TASK_PROMPTS = {
    "zero_shot_classification": """
### ROLE & TASK
You are an expert in spotting misinformation in news articles.
Your job is to decide whether the article is **Credible (1)** or **Fake (0)**, estimate a probability that your decision is correct and provide a reason.

### STEPS
1. First read the ARTICLE. Note any verifiable factual errors.
2. Think about your classification and then output exactly the JSON below.
    """,
    "few_shot_classification": """
### ROLE & TASK
You are an expert in spotting misinformation in news articles.
Your job is to decide whether the article is **Credible (1)** or **Fake (0)**, estimate a probability that your decision is correct and provide a reason.
You are an expert in detecting misinformation in full-length news articles.

### STEPS
1. First read the ARTICLE. Note any verifiable factual errors
2. Use the few-shot examples provided as guidance for how such decisions are typically made.
3. Think in a hidden <!--scratch--> block, then output exactly the JSON below.
""",
    "zero_shot_classification_signals": """
### ROLE
CLASSIFIER: You are an expert at classifying articles as REAL or FAKE.

### TASK CONTEXT
You receive (a) ARTICLE and ARTICLE TITLE and (b) CREDIBILITY SIGNAL classifications from an extractor.
Each credibility signal classification includes:
- label: whether the signal is present (TRUE/FALSE)
- polarity: whether the presence of the signal makes it more credible (POSITIVE) or less credible (NEGATIVE)
- explanation: reasoning for the extractor's classification
- confidence: How certain the extractor is in its classification (LOW/MEDIUM/HIGH)

### TASK STEPS
1. First read the ARTICLE.
2. Then, consult the provided CREDIBILITY SIGNAL classifications
3. Weigh the polarity, confidence, reasoning and label of each signal and combine that with your own reasoning to classify the article as either REAL or FAKE.
4. Referencing the data used in your reasoning, provide a highly rigorous, bullet pointed explanation for you answer.
5. Output in valid JSON, as below:
    """,
    "zero_shot_classification_signals_critic": """
### ROLE
CLASSIFIER: You are an expert at classifying articles as REAL or FAKE.

### TASK CONTEXT
You receive:
(a) ARTICLE and ARTICLE TITLE
(b) CREDIBILITY SIGNAL classifications from an extractor, filter by a critic based on quality and relevance
(c) OVERALL_EXTRACTOR_TRUST: Tells you how consistently, accurately, and reliably the signal-extractor performed across all credibility signals for this article.

Each credibility signal classification includes:
- label: whether the signal is present (TRUE/FALSE)
- polarity: whether the presence of the signal makes it more credible (POSITIVE) or less credible (NEGATIVE)
- quality: how well the extractor explained the classification (LOW/MEDIUM/HIGH)
- relevance: how relevant the presence of the signal is to the overral article classification (LOW/MEDIUM/HIGH)
- explanation: reasoning for the classification

### TASK STEPS
1. First read the ARTICLE and note any verifiable factual errors.
2. Then, consult the provided CREDIBILITY SIGNAL classifications and OVERALL_EXTRACTOR_TRUST score. 
How to use the OVERALL_EXTRACTOR_TRUST score:
- HIGH: treat each signal's quality at face value.
- MEDIUM: treat all quality as approximate; downgrade your reliance on marginal signals.
- LOW: the extractor struggled (signals were ambiguous or errors occurred); use the signals only as weak hints and rely primarily on your direct reading of the article.
3. Weigh the relevance, quality, reasoning and label of each signal and combine that with your own reasoning to classify the article as either REAL or FAKE.
4. Referencing the data used in your reasoning, provide a highly rigorous, bullet pointed explanation for you answer.
5. Output in valid JSON, as below:
    """,
    "few_shot_classification_signals": """
### ROLE & TASK
You are an expert in spotting misinformation in news articles.  
Your job is to decide whether the article is **Credible (1)** or **Fake (0)**, estimate a probability that your decision is correct and provide a reason.

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
### ROLE
CLASSIFIER: You are an expert at analysising news articles and classifying their credibility signals.

### TASK CONTEXT
You receive (a) ARTICLE and ARTICLE TITLE and (b) a list of CREDIBILITY SIGNALS to classify.

### TASK STEPS
For each signal:
1. Read the article and extract multiple passages related to the signal as article_excerpts.
2. Answer TRUE or FALSE to the signal question.
3. Provide a confidence for your answer (LOW/MEDIUM/HIGH).
4. Referencing the extracted passages, provide a rigorous, bullet pointed explanation for you answer.
5. Output in valid JSON, as below:
        """,
    "signal_classification_critic": """
### ROLE
CRITIC: quality-control gate between the signal-extractor and the final REAL/FAKE
classifier.

### INPUT
You receive:
1. **ARTICLE_TOPIC**
2. **CREDIBILITY_SIGNALS** - JSON with one object per signal.  
   Each object has:
    - label: "TRUE|FALSE"  
    - polarity: "POSITIVE|NEGATIVE" 
    - article_excerpts: list of passages from the article that evidence the signal
    - explanation: free-text rationale (may quote the article) 
    - confidence: "LOW|MEDIUM|HIGH"

### YOUR TASK
For **each** signal:
1. **Check the quality of extraction**  
    - Does the explanation really justify the label & polarity?
    - If the explanation shows the *opposite* of the extractor's label
        -> flip the label (`TRUE`<->`FALSE`) and set `was_flipped = true`
        -> set `confidence = "LOW"`
        -> set `quality = "LOW"`
    - If the explanation is unrelated or nonsensical -> keep the original label but set `quality = "LOW"`
    - If the explanation is < 15 words **or** contains no article-specific detail, set `quality = "MEDIUM"` (unless already "LOW")
    - Otherwise set `quality = "HIGH"`

2. **Assess relevance** of the credibility signal as an indicator of article validity based on the ARTICLE_TOPIC: "HIGH/MEDIUM/LOW"

3. **Followup**
    - If the relevance is "MEDIUM" and the signal quality is "LOW" then set `needs_followup = true`
    - If the relevance is "HIGH" and the signal quality is "LOW" or "MEDIUM then set `needs_followup = true`

5. **Decide keep / drop**
   - KEEP if `quality = "HIGH"` **or** (`quality = "MEDIUM"` and `relevance != "LOW"`).
   - otherwise DROP.
6. Output in valid JSON, as below:
        """,
    "followup_analysis": """
You are conducting a detailed follow-up analysis of a credibility signal in an article.
Your task is to:
1. Thoroughly analyze the presence and significance of this signal
2. Evaluate the strength and reliability of the signal
3. Consider any potential false positives or ambiguities
4. Provide a final determination with high confidence
""",
    "critic": """
Evaluate if a model needs to be called. Based on the text 
Return TRUE if confidence is low or signals are unclear.
Return FALSE otherwise.
    """,
}

STRUCTURED_OUTPUTS = {
    "article_classification": """
{
  "label": "REAL|FAKE",
  "confidence": "LOW|MEDIUM|HIGH",
  "explanation": LIST[str]
}
    """,
    "bulk_signals": """
{
  "signals": {
    "<signal_name>": {
      "label": "TRUE|FALSE",
      "confidence": "LOW|MEDIUM|HIGH",
      "article_excerpts": LIST[str],
      "explanation": LIST[str]
    }
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
{
  "reliable_signals": {       // signals to keep
    "<signal_name>": {
      "label": "TRUE|FALSE",
      "confidence": "LOW|MEDIUM|HIGH",
      "polarity": "POSITIVE|NEGATIVE",
      "relevance": "LOW|MEDIUM|HIGH",
      "quality": "LOW|MEDIUM|HIGH",
      "explanation": "<trimmed or improved explanation>"
    },
    ...
  },
  "followup_signals": [             // signals needing re-check
    "<signal_name>", ...
  ],
  "overall_extraction_trust": "LOW|MEDIUM|HIGH",   // LOW if >50 % signals dropped
  "critic_notes": [                 // 2-4 bullets summarising main issues
    "- …",
    "- …"
  ]
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
    "base_inputs": """
### ARTICLE
__Article Title__
{title}
__Article Content__
{content}
### END ARTICLE
""",
    "bulk_signals_extension": """
### CREDIBILITY SIGNAL QUESTIONS
__Signals to detect__
{signals_list}
### END CREDIBILITY SIGNAL QUESTIONS
     """,
    "individual_signal_extension": """
\n
Credibility Signal to detect: {signal_type}
{signal_config[prompt]}
     """,
    "signal_classification_critic_extension": """
### ARTICLE TOPIC 
This article's coarse topic is **{topic}**.
       
### CREDIBILITY SIGNALS
{signals_list}
### END CREDIBILITY SIGNALS
     """,
    "critic_extension": """
### OVERALL_EXTRACTOR_TRUST
The extractor's overall trust score is **{overall_extraction_trust}**.

### CREDIBILITY SIGNALS
The critic's selection of credibility signals are listed below: 
{critic_list}
### END CREDIBILITY SIGNALS
    """,
    "followup_analysis_classification_extension": """
\n
Followup analysis on the credibility signals has been conducted and is output below: 
{followup_analysis} 
    """,
    "classified_signals_extension": """ 
### CREDIBILITY SIGNALS
{signals_list}
### END CREDIBILITY SIGNALS
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
