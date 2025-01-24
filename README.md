# BRAINTEASER: A Novel Task Defying Common Sense

## Project Description

This project is part of the In-Context Learning (ICL) course at the University of Cologne. The focus of this project is to experiment with and compare the performance of [**LLaMA 3.1 & 3.2**](https://llama.meta.com/), [**Phi-3.5-MoE**](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct), and [**Deepseek R1**](https://huggingface.co/deepseek-ai/DeepSeek-R1) models on [**SemEval 2024 Task BRAINTEASER: A Novel Task Defying Common Sense**](https://brainteasersem.github.io/).

### Task Overview

The [SemEval 2024 BRAINTEASER](https://brainteasersem.github.io/) task focuses on testing language models' ability to exhibit lateral thinking and defy default commonsense associations. The challenge includes two main subtasks:

- **Sentence Puzzle**: Brain teasers centered on sentence snippets that require defying commonsense assumptions
- **Word Puzzle**: Brain teasers focused on letter composition where answers violate the default word meanings

Both subtasks include adversarial examples created by manually modifying original brain teasers while preserving their reasoning paths.

## Requirements

To run this project, you need the following:

- Python 3.10 or higher
- Jupyter Notebook
- Required Python libraries listed in `requirements.txt`
- Access to the LLaMA 3.1, LLaMA 3.2, Phi-3.5-MoE, and Deepseek R1 models (instructions provided below)

## Getting Started

### 1. Clone the Repository and Set Up the Environment

1. Clone the repository and navigate to the project directory:

    ```bash
    git clone https://github.com/S0PEX/ICL-BRAINTEASER.git
    cd ICL-BRAINTEASER
    ```

2. Create a virtual environment and activate it:

    - On macOS/Linux:

      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

    - On Windows:

      ```bash
      python -m venv venv
      venv\Scripts\activate
      ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### 2. Download the Models

- **LLaMA 3.1 & 3.2**: Follow the instructions provided [here](https://huggingface.co/meta-llama) to download and set up the LLaMA models
- **Phi-3.5-MoE**: Available at [Hugging Face](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)
- **Deepseek R1**: Available at [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1)

### 3. Prepare the Data

The dataset for this project is available through the [BRAINTEASER Codalab Competition](https://brainteasersem.github.io/). Follow these steps to access and prepare the data:

1. Visit the [BRAINTEASER competition page](https://brainteasersem.github.io/#data)
2. Register for the competition and download the dataset
3. Place the downloaded data in the `data/` directory of your local project folder

### 4. Run Jupyter Notebooks

All experiments are conducted in Jupyter Notebooks. To start the notebook environment, run:

```bash
jupyter notebook
```

This will open a new browser window where you can navigate to the notebook files and run the experiments.
