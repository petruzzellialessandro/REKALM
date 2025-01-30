# Empowering Recommender Systems based on Large Language Models through Knowledge Injection Techniques

![Architecture](img/Architecture.png)

## Table of Contents
1. [Extra Information about Datasets](#extra-information-about-datasets)
2. [Apriori Parameters](#apriori-parameters)
3. [LoRA Hyperparameters](#lora-hyperparameters)
4. [Repository Structure](#repository-structure)
5. [Data Preprocessing](#data-preprocessing)
   - [How to Set Up](#how-to-set-up)

## Extra Information about Datasets
| Dataset       | Users | Items | Ratings | Sparsity  |
|--------------|-------|-------|---------|-----------|
| Last.FM      | 1881  | 2828  | 71,426  | 98.66%    |
| DBbook       | 5660  | 6698  | 129,513 | 99.66%    |
| MovieLens 1M | 6036  | 3081  | 946,120 | 94.91%    |

## Apriori Parameters
| Dataset       | Support  | Confidence | Extracted Rules |
|--------------|----------|------------|-----------------|
| Last.FM      | 0.0015   | 0.002      | 13,391          |
| DBbook       | 0.0003   | 0.001      | 13,245          |
| MovieLens 1M | 0.01     | 0.05       | 62,521          |

## LoRA Hyperparameters
| **Parameter**               | **Value**         |
|-----------------------------|-------------------|
| r                           | 64               |
| alpha                       | 128              |
| target                      | All linear layers |
| sequence length             | 2048             |
| learning rate               | 0.0001           |
| training epochs             | 10               |
| weight decay                | 0.0001           |
| max grad norm               | 1.0              |
| per device train batch size | 4                |
| optimizer                   | AdamW (Torch)    |

## Repository Structure
The repository is organized into three main parts:
- **Data Preprocessing**: Handles both user data preprocessing and knowledge extraction.
- **LLM Processing**: Includes code for fine-tuning and performing inference on the model.
- **Output Refinement and Metrics Calculation**: Evaluates the performance of the recommender system.

## Data Preprocessing
The `DataPreprocessing` folder is structured as follows:
- `data/`: Contains a folder for each dataset along with extra files generated during setup. Each dataset folder includes all necessary files for generating the JSON file required for fine-tuning the model, along with the raw files used to create those JSONs.
- `llama3/`: Contains the Llama 3 tokenizer (tokenizer only), which is used to correctly format the prompt.
- `notebooks/`: Includes all notebooks needed to generate JSON files for fine-tuning the LLM.
- `requirements.txt`: Lists all dependencies required for the project.

### How to Set Up
> **Note:** All necessary files for fine-tuning the model are already in the `data/` folder.

If you wish to generate the files yourself, follow these steps:

1. **Download Raw Text Files**
   - Download the raw text files containing item descriptions from [this link](https://mega.nz/folder/TsMkQaAB#9vxYcaEZhLcr4005L-bbRg).
   - Place the `.txt` files in the corresponding dataset folders.

2. **Set Up the Environment**
   - Create a virtual environment (Python 3.10.12 recommended):
     ```sh
     python -m venv env
     source env/bin/activate  # On Windows use `env\Scripts\activate`
     ```
   - Install the required dependencies:
     ```sh
     pip install -r requirements.txt
     ```

3. **Map DBpedia IDs to Items**
   - Run the `dbpedia_quering.py` script located in the `notebooks/` folder.
   - Ensure you update the dataset selection at the beginning of the script for each dataset.

Once these steps are completed, you can proceed with fine-tuning and evaluation.