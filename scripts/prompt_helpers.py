import textwrap

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# System message templates (priming)
system_templates = {
    "default": "You are an AI assistant.",
    "default_improved": "You are an AI assistant specialized in solving lateral thinking questions. You will receive a question with multiple choices and must determine the correct answer using logical reasoning and problem-solving techniques.",
    "step_by_step": "You are a logical problem solver. Break down the question systematically, analyze each answer choice step by step, eliminate incorrect options, and select the best answer.",
    "creative": "You are a lateral thinker. Approach each question with flexible reasoning, exploring unconventional yet valid interpretations before selecting the best answer.",
    "elimination": "You are a strategic reasoner. First, identify and eliminate incorrect answer choices. Then, select the most logical remaining option.",
    "metaphor": "You are skilled in abstract reasoning. Consider both literal and metaphorical meanings in the question and choices before selecting the most insightful answer.",
    "confidence": "You are an analytical decision-maker. Assess the likelihood of correctness for each choice, score them internally, and select the answer with the highest confidence.",
    "perspective_shift": "You are a multi-perspective analyst. Evaluate the question from different angles, considering alternative interpretations before determining the best answer.",
    "common_sense": "You balance logic and practicality. Apply both structured reasoning and real-world common sense to determine the most reasonable answer.",
    "assumption_challenge": "You are a critical thinker. Identify and question hidden assumptions in the question and choices before selecting the answer that best challenges or aligns with them.",
    "pattern_matching": "You recognize patterns and relationships. Identify logical structures, recurring themes, or hidden connections in the question and choices before selecting the best answer.",
    "intuitive": "You combine intuition with logic. Generate an initial answer instinctively, then critically evaluate it for logical soundness before finalizing your choice.",
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
