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

### 4. Advanced Usage

Performing machine learning experiments can be time-consuming and requires a lot of processing power. To speed up the process, you can use a remote server to offload the processing. The easiest way to do this is to use [**Google Colab**](https://colab.research.google.com/).

A more advanced way to do this is to use a remote server. You can use [**Google Cloud**](https://cloud.google.com/) or [**AWS**](https://aws.amazon.com/) to set up a remote server. Then install the required dependencies and Jupyter Notebook and access it through VSCode or the browser to execute the notebooks. Alternatively, you can extract the notebooks from the Jupyter Notebook and run them in a local Python environment as a script directly.

My current setup is to run the notebooks on a GCP VM and connect to it through VSCode. VSCode will **not** handle uploading the files under `data`, `notebooks`, and `scripts` to the VM. For this you can use `rsync`, `git`, or `scp`. The benefit of using a remote server is that you can leave the VM running and exit from your local VSCode. Therefore, you must spin up a `tmux` or `nohup` session to run the Jupyter Server or the script directly; otherwise, the session will be terminated once you exit from the shell. I prefer to run the Jupyter notebook instead of the script because it is easier to debug and it is more interactive.

Setting up the VM:

```bash
$ sudo apt-get update && sudo apt-get install -y tmux
$ sudo apt-get install -y python3-pip
$ sudo apt-get install -y python3-venv
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ tmux new -s brainteaser
$ jupyter notebook password # set a password for the jupyter notebook server
$ jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 # Then use Ctrl+B, D to detach from the session
```

You can then connect to the Jupyter Notebook server by opening the URL `http://{your_vm_ip}:8888/` in your browser and entering the password you set. Or you can use the VS Code extension to connect to the Jupyter Notebook server.

**Note:** Make sure you open port 8888 in the VM firewall settings and also from the firewall settings in the GCP Console.
