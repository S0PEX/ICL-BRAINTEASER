from pathlib import Path
from dataclasses import dataclass

import numpy as np


@dataclass
class RiddleQuestion:
    """Represents a single riddle question with its answers and metadata"""

    id: str  # Unique identifier for the riddle
    question: str  # The riddle question text
    answer: str  # The correct answer
    distractor1: str  # First incorrect answer choice
    distractor2: str  # Second incorrect answer choice
    distractor_unsure: str  # Third incorrect answer choice (renamed from distractor(unsure) to be valid Python)
    label: int  # Index of the correct answer in choice_list
    choice_list: list[str]  # List of all possible answers
    choice_order: list[int]  # Order of choices as they should be presented


def load_qa_set(file_path: str) -> list[RiddleQuestion]:
    """
    Load riddle questions from a numpy file and convert them to RiddleQuestion objects.

    Args:
        file_path: Path to the .npy file containing riddle data

    Returns:
        List of RiddleQuestion objects
    """
    dataset = np.load(file_path, allow_pickle=True)
    riddle_questions = []

    # Convert 'distractor(unsure)' key to 'distractor_unsure' before creating RiddleQuestion
    # addionally, rename 'distrator1' and 'distrator2' to 'distractor1' and 'distractor2' respectively to fix typos
    field_map = {
        "distractor(unsure)": "distractor_unsure",
        "distrator(unsure)": "distractor_unsure",
        "distrator1": "distractor1",
        "distrator2": "distractor2",
    }
    riddle_questions = [
        RiddleQuestion(**{field_map.get(k, k): v for k, v in item.items()})
        for item in dataset
    ]
    return riddle_questions


class BrainteaserDataset:
    """Manages loading and access to brainteaser datasets including Simple Problems (SP) and Word Problems (WP)"""

    def __init__(self, base_path: Path | str):
        """
        Initialize the dataset manager with the path to dataset files.

        Args:
            base_path: Directory containing the dataset files, can be string or Path

        Raises:
            FileNotFoundError: If base_path doesn't exist
        """
        if isinstance(base_path, str):
            base_path = Path(base_path)

        self.base_path = base_path

        # Ensure the base path exists
        if not self.base_path.exists():
            raise FileNotFoundError(f"The base path {self.base_path} does not exist.")

        # Load SP (Simple Problems) datasets
        self.sp = load_qa_set(f"{self.base_path}/sentence_puzzle.npy")
        self.sp_train = load_qa_set(f"{self.base_path}/SP_train.npy")

        # Load WP (Word Problems) datasets
        self.wp = load_qa_set(f"{self.base_path}/word_puzzle.npy")
        self.wp_train = load_qa_set(f"{self.base_path}/WP_train.npy")
