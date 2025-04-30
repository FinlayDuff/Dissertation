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
You must be thorough and text-grounded, but reveal your reasoning only
inside an HTML comment called <!--scratch-->.

### INPUT
You receive (a) ARTICLE and ARTICLE TITLE and (b) CREDIBILITY SIGNAL classifications from an extractor.
Each credibility signal classification includes:
- label: whether the signal is present (TRUE/FALSE)
- polarity: whether the presence of the signal makes it more credible (POSITIVE) or less credible (NEGATIVE)
- explanation: reasoning for the extractor's classification
- confidence: How certain the extractor is in its classification ("UNCERTAIN|FAIRLY CERTAIN|CERTAIN")

### TASK STEPS
1. Work step-by-step **inside** a <!--scratch--> block:
    a. First read the ARTICLE and attempt to classify the article.
    b. Then, consult the provided CREDIBILITY SIGNAL classifications
    c. Weigh the polarity, confidence, reasoning and label of each signal and combine that with your own reasoning to classify the article as either REAL or FAKE.
    d. Referencing the data used in your reasoning, provide a highly rigorous, bullet pointed explanation for you answer.
2. Output **only** this JSON, wrapped in a ```json fence```, **and nothing else outside**:

    """,
    "zero_shot_classification_signals_critic": """
### ROLE
CLASSIFIER: You are an expert at classifying articles as REAL or FAKE.
You must be thorough and text-grounded, but reveal your reasoning only
inside an HTML comment called <!--scratch-->.

### INPUT
You receive:
(a) ARTICLE and ARTICLE TITLE
(b) CREDIBILITY SIGNAL classifications from an extractor, filtered by a critic based on quality and relevance
(c) OVERALL_EXTRACTOR_TRUST: Tells you how consistently, accurately, and reliably the signal-extractor performed across all credibility signals for this article.

Each credibility signal classification includes:
- label: whether the signal is present (TRUE/FALSE)
- polarity: whether the presence of the signal makes it more credible (POSITIVE) or less credible (NEGATIVE)
- quality: how well the extractor explained the classification (POOR|OK|EXCELLENT)
- relevance: how relevant the presence of the signal is to the overall article classification (LOW/MEDIUM/HIGH)
- explanation: reasoning for the classification

### TASK STEPS
1. Work step-by-step **inside** a <!--scratch--> block:
    a. First read the ARTICLE and note any verifiable factual errors.
    b. Then, consult the provided CREDIBILITY SIGNAL classifications and OVERALL_EXTRACTOR_TRUST score. 
        How to use the OVERALL_EXTRACTOR_TRUST score:
        - HIGH: treat each signal's quality at face value.
        - MEDIUM: treat all quality as approximate; downgrade your reliance on marginal signals.
        - LOW: the extractor struggled (signals were ambiguous or errors occurred); use the signals only as weak hints and rely primarily on your direct reading of the article.
    c. Weigh the relevance, quality, reasoning and label of each signal and combine that with your own reasoning to classify the article as either REAL or FAKE.
    d. Referencing the data used in your reasoning, provide a highly rigorous, bullet pointed explanation for you answer.
2. Output **only** this JSON, wrapped in a ```json fence```, **and nothing else outside**:

    """,
    "zero_shot_classification_signals_critic_followup": """
### ROLE
CLASSIFIER: You are an expert at classifying articles as REAL or FAKE.
You must be thorough and text-grounded, but reveal your reasoning only
inside an HTML comment called <!--scratch-->.

### INPUT
You receive:
(a) ARTICLE and ARTICLE TITLE
(b) CREDIBILITY SIGNAL classifications from an extractor, filtered by a critic based on quality and relevance
(c) OVERALL_EXTRACTOR_TRUST: Tells you how consistently, accurately, and reliably the signal-extractor performed across all credibility signals for this article.
(d) FOLLOWUP_ANALYSIS: A followup analysis of the credibility signal classification that was flagged for follow-up analysis.

Each credibility signal classification includes:
- label: whether the signal is present (TRUE/FALSE)
- polarity: whether the presence of the signal makes it more credible (POSITIVE) or less credible (NEGATIVE)
- quality: how well the extractor explained the classification (POOR|OK|EXCELLENT)
- relevance: how relevant the presence of the signal is to the overall article classification (LOW/MEDIUM/HIGH)
- explanation: reasoning for the classification

### TASK STEPS
1. Work step-by-step **inside** a <!--scratch--> block:
    a. First read the ARTICLE and note any verifiable factual errors.
    b. Then, consult the provided CREDIBILITY SIGNAL classifications and OVERALL_EXTRACTOR_TRUST score. 
        How to use the OVERALL_EXTRACTOR_TRUST score:
        - HIGH: treat each signal's quality at face value.
        - MEDIUM: treat all quality as approximate; downgrade your reliance on marginal signals.
        - LOW: the extractor struggled (signals were ambiguous or errors occurred); use the signals only as weak hints and rely primarily on your direct reading of the article.
    c. Weigh the relevance, quality, reasoning and label of each signal and combine that with your own reasoning to classify the article as either REAL or FAKE.
    d. Referencing the data used in your reasoning, provide a highly rigorous, bullet pointed explanation for you answer.
2. Output **only** this JSON, wrapped in a ```json fence```, **and nothing else outside**:

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

### INPUT
You receive (a) ARTICLE and ARTICLE TITLE and (b) a list of CREDIBILITY SIGNALS to classify.

### TASK STEPS
For each signal:
1. Read the article and extract multiple passages related to the signal as article_excerpts.
2. Answer TRUE or FALSE to the signal question.
3. Provide a confidence for your answer "UNCERTAIN|FAIRLY CERTAIN|CERTAIN".
4. Referencing the extracted passages, provide a rigorous, bullet pointed explanation for you answer.
5. Output **only** this JSON, wrapped in a ```json fence```, **and nothing else outside**:

        """,
    "signal_classification_critic": """
### ROLE
CRITIC — quality-control gate between extractor and final classifier.

### INPUT
- ARTICLE_TOPIC  
- CREDIBILITY_SIGNALS  (label, polarity, explanation, confidence)

### TASK STEPS
For each signal:
1. Check if the explanation supports the label + polarity. Here's a starting point:
   - If it contradicts -> `quality = "POOR"`
   - If unrelated / generic -> `quality = "POOR"`
   - If short (<15 words)   -> `quality = "OK"`
   - Else                   -> `quality = "EXCELLENT"`

2. Based on the topic of the article, assign the relevance of a credibility signal: "HIGH"/"MEDIUM"/"LOW"

3. Decide the pipeline action 
   KEEP         if `quality = "EXCELLENT"`   OR (`quality = "OK"`  AND `relevance is not "LOW"`)  
   FOLLOW_UP    if `relevance = "HIGH" or "MEDIUM"`  AND `quality is not "EXCELLENT"` 
   DROP         if `relevance = "LOW"`   AND `quality is not "EXCELLENT"`
4. Output **only** this JSON, wrapped in a ```json fence```, **and nothing else outside**:

        """,
    "followup_analysis": """
### ROLE
FOLLOW_UP — second-opinion fact-checker for *one* credibility signal.
You must be thorough and text-grounded, but reveal your reasoning only
inside an HTML comment called <!--scratch-->.

### INPUT
You receive 
(a) ARTICLE and ARTICLE TITLE 
(b) CREDIBILITY SIGNAL classification which was flagged for follow-up analysis
(c) CRITIC's explanation of why follow-up is needed

### TASK
1. Work step-by-step **inside** a <!--scratch--> block:
   a. Restate the definition in your own words.
   b. Quote any excerpt lines that relate to the signal.
   c. Decide if the current label is correct; justify.
   d. Re-assign confidence:
           HIGH   - evidence unequivocally supports your label
           MEDIUM - partial / indirect support
           LOW    - evidence sparse or conflicting
   e. Set quality (see below) and draft a public explanation.
2. Output **only** this JSON, wrapped in a ```json fence```, **and nothing else outside**:

""",
    "critic": """
Evaluate if a model needs to be called. Based on the text 
Return TRUE if confidence is low or signals are unclear.
Return FALSE otherwise.
    """,
    "followup_rag": """
### ROLE
Research assistant.

### TASK
Given ARTICLE and TITLE:
1. Extract the **core fact** as a 3-item string:
   "subject - action/claim - key detail (number/date/location)"
2. Produce **3 query variants** that combine:
   • subject + action + key detail (verbatim)
   • subject + action + key detail (synonyms) 
   • subject + key detail in quotes.
3. Output **only** this JSON, wrapped in a ```json fence```, **and nothing else outside**:

    """,
    "followup_corroboration": """
### ROLE
ANALYST: You are a helpful assistant that can summarise articles and decide if the source is credible.
You must be thorough and text-grounded, but reveal your reasoning only
inside an HTML comment called <!--scratch-->.

### INPUT
You receive (a) SOURCE ARTICLE TITLE (b) the CORE FACT and (c) a list of RETRIEVED ARTICLES.

### TASK
For each RETRIEVED ARTICLE, you will be given a title, body and href. 
1. Work step-by-step **inside** a <!--scratch--> block:
    a. Summarise the article in 5 bullet points.
    b. Decide if the article describes the same CORE FACT of the soure article. If it does, store the title and href.
    c. Decide if the source is reputable.
2. Decide whether the source artical is corroborated by a reputable source and output your reasoning.
3. Output **only** this JSON, wrapped in a ```json fence```, **and nothing else outside**:

    """,
    "followup_claim_extractor": """
### ROLE
You are a helpful assistant that can extract the main claims from an article.
You must be thorough and text-grounded, but reveal your reasoning only
inside an HTML comment called <!--scratch-->.

### TASK
1. Work step-by-step **inside** a <!--scratch--> block:
    a. You will be given an ARTICLE and its title. Your task is to find the main claims the article makes.
    b. For each claim extracted:
        - Directly quote the claim from the article.
        - Quote the evidence presented in the article to support the claim.
        - Assign an importance score to the claim based on how important it is to the article (LOW/MEDIUM/HIGH).
    c. Select up to 3 of the most important claims and output them in a list, ensuring they are distinct and not duplicates.
3. Output **only** this JSON, wrapped in a ```json fence```, **and nothing else outside**:

    """,
    "followup_claim_verification": """
### ROLE
ANALYST: You are a helpful assistant that can verify claims.
You must be thorough and text-grounded, but reveal your reasoning only
inside an HTML comment called <!--scratch-->.

### INPUT
You receive a list of CLAIMS made in an article.
For claim you will be given:
- claim: the claim made in the article
- importance: the importance of the claim to the article
- evidence: evidence supporting the claim
- retrieved_articles: a list of articles/websites that may confirm the claim

### TASK
For each CLAIM:
1. Work step-by-step **inside** a <!--scratch--> block:
    a. For each CLAIM:
        - Read the articles retrieved related to the claim and decide whether they corroborate the claim and are reputable.
        - Read the claim, the evidence the article provided and the retrieved articles.
        - Based on this, decide if the claim is explicitly unverified or lacks confirmation by reliable sources.
    b. Based on the claims identified and their respective 'importance', decide if the text contain claims that are explicitly unverified or lack confirmation by reliable sources?
    c. Summarise your scratch reasoning and provide a highly rigorous, bullet pointed explanation for you answer.
2. Output **only** this JSON, wrapped in a ```json fence```, **and nothing else outside**:

    """,
    "feature_selector": """
### ROLE: Research assistant.
You are a helpful assistant that can merge the output of multiple stages of analysis.

### INPUT
You will receive a JSON object containing:
(a) credibility_signals: a list of credibility signals that have been extracted, each with:
    - "label": "TRUE|FALSE",
    - "confidence": "UNCERTAIN|FAIRLY CERTAIN|CERTAIN",
    - "article_excerpts": LIST[str],
    - "explanation": LIST[str]
(b) critic_output: critique of these extracted signals:
    1. "keep": a list of signals to keep, each with:
        - "label": "TRUE|FALSE",
        - "confidence": "UNCERTAIN|FAIRLY CERTAIN|CERTAIN",
        - "polarity": "POSITIVE|NEGATIVE",
        - "relevance": "The critic's assessment of the relevance of the signal to the article classification"
        - "quality": "The critic's assessment of how accurate the classification of the signal is"
        - "explanation": "The critic's explanation for keeping the signal"
    2. "follow_up": a list of signals flagged for follow-up analysis
    3. "critic_notes": the reasoning for the critics overall decisions
    4. "overall_trust": the critic's overall trust in the extractor
(c) followup_signals_analysis: a list of follow-up analyses, each with:
    - "label": "TRUE|FALSE",each with a label (TRUE/FALSE)
    - "analysis_type": (LLM/RAG/Classifier)
    - "explanation": Optional: reasoning for the classification

### TASK
1. Work step-by-step **inside** a <!--scratch--> block
    a. Read the article and the json object.
    b. For each **signal**:
        i) Read the original classification, the critics output and the subsequent follow-up analysis
        ii) Combine the output of each stage and determine the final label for the signal (TRUE|FALSE)
        iii) Assign a confidence to your classification (UNCERTAIN|FAIRLY CERTAIN|CERTAIN)
        iv) Summarise your scratch reasoning and provide a highly rigorous, bullet pointed explanation for you answer.
2. Output **only** this JSON, wrapped in a ```json fence```, **and nothing else outside**:

    """,
    "zero_shot_classification_feature_selector": """
### ROLE
CLASSIFIER: You are an expert at classifying articles as REAL or FAKE.
You must be thorough and text-grounded, but reveal your reasoning only
inside an HTML comment called <!--scratch-->.

### INPUT
You receive: 
(a) ARTICLE and ARTICLE TITLE
(b) CREDIBILITY SIGNALS that have been extract by a research assistant, each with:
- question: the question the signal is asking
- label: answer to the question (TRUE/FALSE)
- confidence: confidence of the research assistant in the answer "UNCERTAIN|FAIRLY CERTAIN|CERTAIN"
- explanation: reasoning for the extractor's classification of the signal for this article

### TASK STEPS
1. Work step-by-step **inside** a <!--scratch--> block:
    a. First read the ARTICLE create an initial classification of the article (REAL|FAKE)
    b. Then, read the provided CREDIBILITY SIGNAL classifications.
    c. Intepret the importance of each signal as it relates to the article classification, by weighing the confidence, explanation and label of each signal.
    d. Combine this information with your initial classification.
    e. Determine your *final* classification for the article with an associated confidence (UNCERTAIN|FAIRLY CERTAIN|CERTAIN).
    d. Summarise your scratch reasoning and provide a highly rigorous, bullet pointed explanation for you answer.
2. Output **only** this JSON, wrapped in a ```json fence```, **and nothing else outside**:

""",
}

STRUCTURED_OUTPUTS = {
    "article_classification": """
{
  "label": "REAL|FAKE",
  "confidence": "UNCERTAIN|FAIRLY CERTAIN|CERTAIN",
  "explanation": LIST[str]
}
    """,
    "bulk_signals": """
{
  "signals": {
    "<signal_name>": {
      "label": "TRUE|FALSE",
      "confidence": "UNCERTAIN|FAIRLY CERTAIN|CERTAIN",
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
  "keep": {
      "<signal_name>": {
         "label": "TRUE|FALSE",
         "confidence": ""UNCERTAIN|FAIRLY CERTAIN|CERTAIN"",
         "polarity": "POSITIVE|NEGATIVE",
         "relevance": "LOW|MEDIUM|HIGH",
         "quality": "POOR|OK|EXCELLENT",
         "explanation": "<trimmed explanation>"
      },
      …
  },
  "follow_up": {
      "<signal_name>": "why follow-up is needed",
      …
  },
  "overall_trust": "LOW|MEDIUM|HIGH",   // LOW if >50 % of signals drop or flip
  "critic_notes": [
      "- …",
      "- …"
  ]
}
        """,
    "followup_analysis": """
 {
    "label": "TRUE|FALSE",
    "confidence": ""UNCERTAIN|FAIRLY CERTAIN|CERTAIN"",
    "article_excerpts": LIST[str],
    "explanation": LIST[str]
}
            """,
    "followup_rag": """
{
  "core_fact": "...",
  "queries": ["...", "...", "..."],
}
    """,
    "followup_corroboration": """
{
    "corroborating_articles": [
        {
            "title": "Title of the corroborating article",
            "href": "URL of the corroborating article",
            credibility: "Credible|Not Credible"
        },
    ],
    "reputable_corroboration_of_source": "TRUE|FALSE",
    "explanation": ["A detailed explanation of the corroboration or lack thereof of the source article."]
}
    """,
    "followup_claim_extractor": """
[
    {
        "claim": "Claim made in the article",
        "importance": "LOW|MEDIUM|HIGH",
        "evidence": "Evidence supporting the claim"
    },
]
    """,
    "followup_claim_verification": """
{
"claims":
    [
        {
            "claim": "Claim made in the article",
            "importance": "LOW|MEDIUM|HIGH",
            "verified": "TRUE|FALSE",
            "explanation": "Explanation of the verification process and results",
        },
    ],
"explicitly_unverified_claims": "TRUE|FALSE",
"explanation": [
    "A detailed explanation of the verification process and results."
]
"
}
    """,
    "feature_selector": """
{
    "signals": {
        "<signal_name>": {
            "label": "TRUE|FALSE",
            "confidence": "UNCERTAIN|FAIRLY CERTAIN|CERTAIN",
            "explanation": ["<rigourous explanation>"]
        }
    },
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
### FOLLOWUP ANALYSIS
{followup_analysis}
### END FOLLOWUP ANALYSIS
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
    "followup_corroboration": """
### SOURCE ARTICLE
{title}

### CORE FACT
{core_fact}

### RETRIEVED ARTICLES
{retrieved_articles}
### END RETRIEVED ARTICLES
    """,
    "followup_claim_verification": """
### CLAIMS
{claims_list}
### END CLAIMS
""",
    "feature_selector_extension": """
### CREDIBILITY SIGNALS
{signals_list}
### END CREDIBILITY SIGNALS

### CRITIC
{critic_notes}
### END CRITIC

### FOLLOWUP ANALYSIS
{followup_analysis}
### END FOLLOWUP ANALYSIS
""",
    "classification_selected_features_extension": """
### CREDIBILITY SIGNALS
{selected_features}
### END CREDIBILITY SIGNALS
""",
}
