# BRAINTEASER: A Novel Task Defying Common Sense

## Project Description

This project is part of the In-Context Learning (ICL) course at the University of Cologne. The focus of this project is to experiment with and compare the performance of [**Llama 3**](https://ollama.com/library/llama3), [**Phi**](https://ollama.com/library/phi), [**Qwen**](https://ollama.com/library/qwen), [**Gemma**](https://ollama.com/library/gemma), and [**Mistral**](https://ollama.com/library/mistral) models on [**SemEval 2024 Task BRAINTEASER: A Novel Task Defying Common Sense**](https://brainteasersem.github.io/).

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
- Ollama installed for running the models (instructions provided below)

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

All experiments are run using Ollama's compatible API. Ollama is a platform that provides a user-friendly interface for accessing and running various language models. To set up Ollama and download these models:

1. **Install Ollama**: Follow the instructions on [ollama.com](https://ollama.com/) to install Ollama on your machine. This typically involves downloading the installer for your operating system and following the setup instructions.

**Note**: There is no need to manually pull the models, as the executor will take care of it automatically.

Current the following models are used:

- Llama 3.1 (8B)
- Llama 3.2 (1B, 3B)
- Phi 3.5 (3.8B)
- Phi 4 (14B)
- Qwen 2.5 (0.5B, 1.5B, 3B, 7B, 14B, 32B)
- Gemma 2 (2B, 9B, 27B)
- Mistral Nemo (12B)

### 3. Prepare the Data

The dataset for this project is available through the [BRAINTEASER Codalab Competition](https://brainteasersem.github.io/).

Follow these steps to access and prepare the data:

1. Visit the [BRAINTEASER competition page](https://brainteasersem.github.io/#data)
2. Register for the competition and download the dataset
3. Place the downloaded data in the `data/` directory of your local project folder

### 4. Run Jupyter Notebooks

All experiments are conducted in Jupyter Notebooks. We use Ollama to access the models as it provides an OpenAPI compatible API and integrates with LangChain nicely. Additionally, Ollama provides quantized variants of the models, offering advantages over alternatives like vLLM in terms of memory efficiency and performance.

To run the Ollama server concurrently with the Jupyter Notebook, you can use screen, tmux, or a daemon. Alternatively, you can run it in the background using `&` in Linux.

```shell
# Serve Ollama API
$ ollama serve &
# Start Jupyter Notebook Server
$ jupyter notebook
```
