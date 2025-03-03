import random
import textwrap
from typing import Literal
from collections.abc import Callable

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
)

from scripts.dataset import RiddleQuestion
from scripts.executor import Dataset

# System message templates (priming)
system_templates: dict[str, str] = {
    "default": "You are an AI assistant.",
    "default-improved": "You are an AI assistant specialized in lateral thinking puzzles and brain teasers. Analyze each question carefully to find the correct answer among the choices.",
    "step-by-step": "You are a methodical problem solver. For each lateral thinking question, think step by step: first analyze the problem statement, then evaluate each choice systematically, and finally select the single best answer. Show your reasoning process clearly.",
    "creative": "You are a creative puzzle solver. Think step by step beyond conventional reasoning when examining these lateral thinking questions. Consider unexpected connections before selecting your answer.",
    "elimination": "You are a strategic puzzle solver. For each lateral thinking question, think step by step and methodically eliminate implausible choices until you identify the single correct answer.",
    "metaphor": "You are skilled in abstract reasoning. Think step by step to consider hidden meanings and metaphorical interpretations in these brain teasers before selecting the most appropriate choice.",
    "confidence": "You are a precise decision-maker. Think step by step to evaluate each option's likelihood of being correct for these lateral thinking questions and select the single best answer.",
    "perspective-shift": "You are an adaptive thinker. Think step by step to approach each brain teaser from multiple perspectives, challenging your initial assumptions before selecting the correct choice.",
    "common-sense": "You are balanced in logic and practicality when solving puzzles. Think step by step, applying careful reasoning while considering practical implications to find the correct answer.",
    "assumption-challenge": "You are skilled at identifying hidden assumptions in puzzles. Think step by step to question the premises behind each lateral thinking question before selecting your answer.",
    "pattern-matching": "You excel at recognizing patterns in complex problems. Think step by step to identify logical structures and connections in these brain teasers to determine the correct choice.",
    "intuitive": "You combine quick intuition with careful verification. For each puzzle, think step by step by forming an initial impression, then analytically confirm whether your instinct leads to the correct answer.",
}

TemplateNameType = Literal[
    "default",
    "default-improved",
    "step-by-step",
    "creative",
    "elimination",
    "metaphor",
    "confidence",
    "perspective-shift",
    "common-sense",
    "assumption-challenge",
    "pattern-matching",
    "intuitive",
]


def get_system_prompt(template_name: TemplateNameType) -> SystemMessagePromptTemplate:
    system_prompt = system_templates[template_name]
    system_prompt = textwrap.dedent(system_prompt)

    system_prompt_template = SystemMessagePromptTemplate.from_template(
        system_prompt, id=template_name
    )
    return system_prompt_template


def get_user_prompt() -> HumanMessagePromptTemplate:
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
    system_prompt_template_name: TemplateNameType = "default",
) -> ChatPromptTemplate:
    """
    Creates a chat prompt template with system and user prompts.

    Args:
        system_prompt_template_name: The name of the system prompt template to use.
            Defaults to "default".

    Returns:
        A ChatPromptTemplate containing both system and user prompts.
    """
    system_prompt_template = get_system_prompt(system_prompt_template_name)
    user_prompt_template = get_user_prompt()
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_prompt_template, user_prompt_template]
    )

    return chat_prompt_template


def get_few_shot_dataset(
    dataset: Dataset, number_of_shots: int = 4
) -> tuple[list[RiddleQuestion], list[RiddleQuestion]]:
    """
    Splits a dataset into examples for few-shot learning and riddles to solve.

    Args:
        dataset: The dataset containing riddles.
        number_of_shots: Number of examples to use for few-shot learning. Defaults to 4.

    Returns:
        A tuple containing (examples_for_few_shot_learning, riddles_to_solve).
    """
    riddles_as_examples = dataset.riddles[:number_of_shots]
    riddles_to_solve = dataset.riddles[number_of_shots:]
    return (riddles_as_examples, riddles_to_solve)


def get_few_shot_chat_template(
    examples: list[RiddleQuestion],
    args_generator: Callable[[RiddleQuestion], dict],
    system_prompt_template_name: TemplateNameType,
    number_of_shots: int = 4,
) -> ChatPromptTemplate:
    """
    Creates a few-shot learning chat template with examples and returns riddles to solve.

    Args:
        examples: The list of examples to use for few-shot learning.
        args_generator: Function to convert a riddle question into template arguments.
        system_prompt_template_name: The name of the system prompt template to use.
        number_of_shots: Number of examples to use for few-shot learning. Defaults to 4.

    Returns:
        A ChatPromptTemplate containing the few-shot learning template.
    """

    if len(examples) < number_of_shots:
        raise ValueError(
            f"Number of examples ({len(examples)}) must be greater than or equal to the number of shots ({number_of_shots})."
        )

    # Create example prompt template for few-shot learning
    example_prompt = ChatPromptTemplate.from_messages(
        [
            get_user_prompt(),
            ("ai", "{answer}"),
        ]
    )

    # Try to get unique examples by label first
    unique_by_label = {}
    for example in examples:
        if example.label not in unique_by_label:
            unique_by_label[example.label] = example

    # If we have enough unique examples, use them
    if len(unique_by_label) >= number_of_shots:
        riddles_as_examples = list(unique_by_label.values())[:number_of_shots]
    else:
        # Not enough unique labels, so take all unique examples we have
        unique_examples = list(unique_by_label.values())
        # Then randomly sample from remaining examples to reach the desired number
        # Exclude examples that are already in our unique set
        remaining_examples = [ex for ex in examples if ex not in unique_examples]
        random.shuffle(remaining_examples)
        riddles_as_examples = (
            unique_examples
            + remaining_examples[: number_of_shots - len(unique_examples)]
        )

    # Create few-shot prompt with examples
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=[args_generator(example) for example in riddles_as_examples],
    )

    # Combine system prompt, few-shot examples, and user prompt
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            get_system_prompt(system_prompt_template_name),
            few_shot_prompt,
            get_user_prompt(),
        ]
    )

    return chat_prompt_template
