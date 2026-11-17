"""Prompt template library for diverse Q&A generation.

Two template families:
  - PROMPT_TEMPLATES: original single-chunk templates (fallback when RAG is disabled)
  - RAG_PROMPT_TEMPLATES: multi-evidence templates using retrieved context.
    Evidence chunks are provided for factual grounding, but the LLM is instructed
    to produce self-contained answers WITHOUT citation markers.  Source traceability
    is handled at the metadata level (chunk IDs, retrieval scores) rather than
    polluting the generated text with [Evidence N] artefacts.
"""


PROMPT_TEMPLATES: dict[str, str] = {
    "factual": """You are a medical domain expert creating a standalone medical Q&A dataset entry about "{domain}".

Use the following reference material to create ONE factual question-answer pair. The answer must be **completely self-contained** — a reader must be able to understand it without access to any source material.

REFERENCE MATERIAL:
{text}

CRITICAL RULES:
- The answer must read as an independent, authoritative medical statement.
- NEVER use phrases like "according to the text", "the text states", "based on the passage", "as mentioned", "the study found", "the article describes", or any reference to source material.
- Write the answer as if you are a medical textbook author stating established medical knowledge.
- Use professional, unambiguous medical terminology suitable for a clinician audience.
- Be clear, concise, and accurate. Do not speculate or add information not supported by the reference material.
- Difficulty level: {difficulty}

Output Format (exactly as shown):
Question: [Write one high-quality, factual medical question]
Answer: [Write a complete, self-contained answer that stands alone as medical knowledge]

Generate only one QA pair. Do not include any introductory or closing remarks.""",

    "reasoning": """You are a medical domain expert creating a standalone medical Q&A dataset entry about "{domain}".

Use the following reference material to create ONE question that requires clinical reasoning. The answer must be **completely self-contained** — a reader must be able to understand it without access to any source material.

REFERENCE MATERIAL:
{text}

CRITICAL RULES:
- The question should start with "Why", "How does", "What mechanism", or similar reasoning prompts.
- The answer must explain the reasoning chain as an independent, authoritative medical explanation.
- NEVER use phrases like "according to the text", "the text states", "based on the passage", "as mentioned", "the study found", "the article describes", or any reference to source material.
- Write the answer as if you are a medical textbook author explaining clinical reasoning.
- Use professional medical terminology suitable for a clinician audience.
- Difficulty level: {difficulty}

Output Format (exactly as shown):
Question: [Write one question requiring clinical reasoning]
Answer: [Write a detailed, self-contained answer explaining the reasoning as established medical knowledge]

Generate only one QA pair. Do not include any introductory or closing remarks.""",

    "comparison": """You are a medical domain expert creating a standalone medical Q&A dataset entry about "{domain}".

Use the following reference material to create ONE comparison question and answer. The answer must be **completely self-contained** — a reader must be able to understand it without access to any source material.

REFERENCE MATERIAL:
{text}

CRITICAL RULES:
- Formulate a question that compares two or more conditions, treatments, drugs, or approaches.
- Use phrases like "How does X compare to Y", "What are the differences between", "Which is more effective".
- The answer must clearly outline similarities and differences as an authoritative medical comparison.
- NEVER use phrases like "according to the text", "the text states", "based on the passage", "as mentioned", "the study found", "the article describes", or any reference to source material.
- Write the answer as if you are a medical textbook author presenting a comparison.
- Use professional medical terminology suitable for a clinician audience.
- Difficulty level: {difficulty}

Output Format (exactly as shown):
Question: [Write one comparison question]
Answer: [Write a structured, self-contained comparison answer presenting established medical knowledge]

Generate only one QA pair. Do not include any introductory or closing remarks.""",

    "application": """You are a medical domain expert creating a standalone medical Q&A dataset entry about "{domain}".

Use the following reference material to create ONE clinical application/scenario question and answer. The answer must be **completely self-contained** — a reader must be able to understand it without access to any source material.

REFERENCE MATERIAL:
{text}

CRITICAL RULES:
- Create a realistic clinical scenario question (e.g., "A patient presents with X, what is the recommended approach?").
- The answer must describe the appropriate clinical action as an authoritative medical recommendation.
- Focus on practical clinical decision-making: diagnosis, treatment selection, management.
- NEVER use phrases like "according to the text", "the text states", "based on the passage", "as mentioned", "the study found", "the article describes", or any reference to source material.
- Write the answer as if you are a medical textbook author providing clinical guidance.
- Use professional medical terminology suitable for a clinician audience.
- Difficulty level: {difficulty}

Output Format (exactly as shown):
Question: [Write one clinical scenario/application question]
Answer: [Write a practical, self-contained clinical answer presenting established medical guidance]

Generate only one QA pair. Do not include any introductory or closing remarks.""",
}


# Difficulty descriptors appended to prompts
DIFFICULTY_DESCRIPTORS: dict[str, str] = {
    "beginner": "Beginner — suitable for medical students; use accessible language and focus on fundamental concepts.",
    "intermediate": "Intermediate — suitable for residents and general practitioners; use standard clinical terminology.",
    "advanced": "Advanced — suitable for specialists; use specialized terminology and address nuanced clinical details.",
}


def build_prompt(
    template_name: str,
    text: str,
    domain: str,
    difficulty: str = "intermediate",
) -> str:
    """Build a complete prompt from a template name, text chunk, domain, and difficulty."""
    template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["factual"])
    difficulty_desc = DIFFICULTY_DESCRIPTORS.get(difficulty, DIFFICULTY_DESCRIPTORS["intermediate"])
    return template.format(domain=domain, text=text, difficulty=difficulty_desc)


# ── RAG-aware prompt templates ─────────────────────────────────────────
#
# These templates take formatted evidence passages from the retriever and
# ask for a scoped, standalone answer or "NOT FOUND".
#

_RAG_SHARED_RULES = """CRITICAL RULES:
- You may ONLY use information present in the EVIDENCE blocks above.
- The answer must be **completely self-contained** — a reader must be able to understand it without access to any source material or evidence blocks.
- NEVER include citation markers like [Evidence 1], [Evidence 2], [1], [2], or any bracketed references in your answer.
- Preserve the study population, setting, measured outcome, and uncertainty whenever they qualify a finding. Scope the question to the same population or setting when relevant.
- Make the answer readable on its own. Identify a finding's study design or population explicitly, without referring to an unnamed passage or study.
- Preserve the distinction between association and causation, and between an inconclusive result and evidence of no effect.
- If the evidence is insufficient or materially conflicting for the proposed answer, output EXACTLY: NOT FOUND
- Do not extrapolate a single study finding to general medical guidance.
- Difficulty level: {difficulty}"""

RAG_PROMPT_TEMPLATES: dict[str, str] = {
    "factual": """You are a medical domain expert creating a standalone medical Q&A dataset entry about "{domain}".

Below are retrieved evidence passages relevant to this topic. Use them to create ONE factual question-answer pair.

RETRIEVED EVIDENCE:
{evidence}

""" + _RAG_SHARED_RULES + """

Output Format (exactly as shown):
Question: [Write one high-quality, factual medical question]
Answer: [Write a complete, self-contained answer within the scope of the evidence]

Generate only one QA pair. Do not include any introductory or closing remarks.""",

    "reasoning": """You are a medical domain expert creating a standalone medical Q&A dataset entry about "{domain}".

Below are retrieved evidence passages relevant to this topic. Use them to create ONE question that requires clinical reasoning.

RETRIEVED EVIDENCE:
{evidence}

""" + _RAG_SHARED_RULES + """
- The question should start with "Why", "How does", "What mechanism", or similar reasoning prompts.
- The answer must explain the reasoning supported by the evidence, preserving uncertainty about mechanisms.

Output Format (exactly as shown):
Question: [Write one question requiring clinical reasoning]
Answer: [Write a self-contained answer explaining the supported reasoning]

Generate only one QA pair. Do not include any introductory or closing remarks.""",

    "comparison": """You are a medical domain expert creating a standalone medical Q&A dataset entry about "{domain}".

Below are retrieved evidence passages relevant to this topic. Use them to create ONE comparison question.

RETRIEVED EVIDENCE:
{evidence}

""" + _RAG_SHARED_RULES + """
- Formulate a question that compares two or more conditions, treatments, drugs, or approaches.
- Use phrases like "How does X compare to Y", "What are the differences between", "Which is more effective".
- The answer must preserve which populations, comparisons, and outcomes the evidence actually addresses.

Output Format (exactly as shown):
Question: [Write one comparison question]
Answer: [Write a self-contained comparison supported by the provided evidence]

Generate only one QA pair. Do not include any introductory or closing remarks.""",

    "application": """You are a medical domain expert creating a standalone medical Q&A dataset entry about "{domain}".

Below are retrieved evidence passages relevant to this topic. Use them to create ONE clinical application/scenario question.

RETRIEVED EVIDENCE:
{evidence}

""" + _RAG_SHARED_RULES + """
- Create a realistic clinical scenario question (e.g., "A patient presents with X, what is the recommended approach?").
- Use this question type only if the passages explicitly contain a recommendation for the described scenario and population. Otherwise output EXACTLY: NOT FOUND

Output Format (exactly as shown):
Question: [Write one clinical scenario/application question]
Answer: [Summarise the supported recommendation and its stated conditions]

Generate only one QA pair. Do not include any introductory or closing remarks.""",
}


def build_rag_prompt(
    template_name: str,
    evidence_text: str,
    domain: str,
    difficulty: str = "intermediate",
) -> str:
    """Build a RAG-aware prompt with retrieved evidence.

    Parameters
    ----------
    template_name : one of factual, reasoning, comparison, application
    evidence_text : pre-formatted evidence block from RetrievalResult.format_context()
    domain : the medical domain / topic
    difficulty : beginner, intermediate, or advanced
    """
    template = RAG_PROMPT_TEMPLATES.get(template_name, RAG_PROMPT_TEMPLATES["factual"])
    difficulty_desc = DIFFICULTY_DESCRIPTORS.get(difficulty, DIFFICULTY_DESCRIPTORS["intermediate"])
    return template.format(domain=domain, evidence=evidence_text, difficulty=difficulty_desc)
