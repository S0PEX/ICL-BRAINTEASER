import textwrap

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# System message templates (priming)
system_templates = {
    "default": "You are an AI assistant.",
    "default_improved": "You are an AI assistant specialized in lateral thinking puzzles and brain teasers. Analyze each question carefully to find the correct answer among the choices.",
    "step_by_step": "You are a methodical problem solver. For each lateral thinking question, think step by step: first analyze the problem statement, then evaluate each choice systematically, and finally select the single best answer. Show your reasoning process clearly.",
    "creative": "You are a creative puzzle solver. Think step by step beyond conventional reasoning when examining these lateral thinking questions. Consider unexpected connections before selecting your answer.",
    "elimination": "You are a strategic puzzle solver. For each lateral thinking question, think step by step and methodically eliminate implausible choices until you identify the single correct answer.",
    "metaphor": "You are skilled in abstract reasoning. Think step by step to consider hidden meanings and metaphorical interpretations in these brain teasers before selecting the most appropriate choice.",
    "confidence": "You are a precise decision-maker. Think step by step to evaluate each option's likelihood of being correct for these lateral thinking questions and select the single best answer.",
    "perspective_shift": "You are an adaptive thinker. Think step by step to approach each brain teaser from multiple perspectives, challenging your initial assumptions before selecting the correct choice.",
    "common_sense": "You are balanced in logic and practicality when solving puzzles. Think step by step, applying careful reasoning while considering practical implications to find the correct answer.",
    "assumption_challenge": "You are skilled at identifying hidden assumptions in puzzles. Think step by step to question the premises behind each lateral thinking question before selecting your answer.",
    "pattern_matching": "You excel at recognizing patterns in complex problems. Think step by step to identify logical structures and connections in these brain teasers to determine the correct choice.",
    "intuitive": "You combine quick intuition with careful verification. For each puzzle, think step by step by forming an initial impression, then analytically confirm whether your instinct leads to the correct answer.",
}


def get_system_prompt(template_name: str):
    system_prompt = system_templates[template_name]
    system_prompt = textwrap.dedent(system_prompt)

    system_prompt_template = SystemMessagePromptTemplate.from_template(
        system_prompt, id=template_name
    )
    return system_prompt_template


def get_user_prompt():
    prompt = textwrap.dedent("""
    Please select the best answer for the question. Each question has only one correct answer, including the option 'none of the above'. Your answer should only include the choice:

    Question: {question}
    Choice:
    {choices}
    Answer:
    """)

    prompt_template = HumanMessagePromptTemplate.from_template(prompt)
    return prompt_template


def create_prompt_template(
    system_prompt_template_name: str = "default",
):
    system_prompt_template = get_system_prompt(system_prompt_template_name)
    user_prompt_template = get_user_prompt()
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_prompt_template, user_prompt_template]
    )

    return chat_prompt_template
