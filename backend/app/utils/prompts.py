"""Prompt template library for diverse Q&A generation."""


PROMPT_TEMPLATES: dict[str, str] = {
    "factual": """You are a medical domain expert and NLP assistant trained to generate high-quality, clinically relevant Question-Answer (QA) pairs from medical literature.

Your task:
Given the following medical text related to "{domain}", generate exactly **ONE** factual question and its precise, evidence-based answer, using only the information explicitly stated in the text.

TEXT:
{text}

Instructions:
- Formulate a factual question about specific medical facts (e.g., definitions, statistics, mechanisms, classifications).
- Use professional, unambiguous medical terminology suitable for a clinician audience.
- Ensure the answer is **strictly derived from the text**, without introducing outside knowledge.
- Be clear, concise, and accurate. Do not speculate.
- Difficulty level: {difficulty}

Output Format (exactly as shown):
Question: [Write one high-quality, factual medical question]
Answer: [Write a complete, well-supported answer based only on the given text]

Generate only one QA pair. Do not include any introductory or closing remarks.""",

    "reasoning": """You are a medical domain expert and NLP assistant trained to generate high-quality, clinically relevant Question-Answer (QA) pairs from medical literature.

Your task:
Given the following medical text related to "{domain}", generate exactly **ONE** question that requires clinical reasoning to answer, using only the information in the text.

TEXT:
{text}

Instructions:
- Formulate a question that requires understanding of cause-effect relationships, pathophysiology, or treatment rationale.
- The question should start with "Why", "How does", "What mechanism", or similar reasoning prompts.
- Ensure the answer explains the reasoning chain, not just states a fact.
- Use professional medical terminology suitable for a clinician audience.
- Answer must be **strictly derived from the text**.
- Difficulty level: {difficulty}

Output Format (exactly as shown):
Question: [Write one question requiring clinical reasoning]
Answer: [Write a detailed answer explaining the reasoning based on the text]

Generate only one QA pair. Do not include any introductory or closing remarks.""",

    "comparison": """You are a medical domain expert and NLP assistant trained to generate high-quality, clinically relevant Question-Answer (QA) pairs from medical literature.

Your task:
Given the following medical text related to "{domain}", generate exactly **ONE** comparison question and answer, using only the information in the text.

TEXT:
{text}

Instructions:
- Formulate a question that compares two or more conditions, treatments, drugs, or approaches mentioned in the text.
- Use phrases like "How does X compare to Y", "What are the differences between", "Which is more effective".
- The answer should clearly outline similarities and differences based on the text.
- Use professional medical terminology suitable for a clinician audience.
- Answer must be **strictly derived from the text**.
- Difficulty level: {difficulty}

Output Format (exactly as shown):
Question: [Write one comparison question]
Answer: [Write a structured comparison answer based on the text]

Generate only one QA pair. Do not include any introductory or closing remarks.""",

    "application": """You are a medical domain expert and NLP assistant trained to generate high-quality, clinically relevant Question-Answer (QA) pairs from medical literature.

Your task:
Given the following medical text related to "{domain}", generate exactly **ONE** clinical application/scenario question and answer, using only the information in the text.

TEXT:
{text}

Instructions:
- Create a realistic clinical scenario question (e.g., "A patient presents with X, what is the recommended approach?").
- The answer should describe the appropriate clinical action based on the text.
- Focus on practical clinical decision-making: diagnosis, treatment selection, management.
- Use professional medical terminology suitable for a clinician audience.
- Answer must be **strictly derived from the text**.
- Difficulty level: {difficulty}

Output Format (exactly as shown):
Question: [Write one clinical scenario/application question]
Answer: [Write a practical clinical answer based on the text]

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
