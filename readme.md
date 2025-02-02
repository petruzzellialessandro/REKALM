# Empowering Recommender Systems based on Large Language Models through Knowledge Injection Techniques

![Architecture](img/Architecture.png)

## Table of Contents
1. [Abstract](#abstract)
2. [Datasets Information](#datasets-information)
3. [Apriori Algorithm Parameters](#apriori-algorithm-parameters)
4. [LoRA Hyperparameters](#lora-hyperparameters)
5. [Repository Structure](#repository-structure)
6. [Data Preprocessing](#data-preprocessing)
   - [Setting Up the Environment](#setting-up-the-environment)
   - [Generating Training Data](#generating-training-data)
7. [Large Language Model (LLM) Training and Inference](#large-language-model-llm-training-and-inference)
   - [Creating the Singularity Container](#creating-the-singularity-container)
   - [Training the Model](#training-the-model)
   - [Merging the Adapter with the Base Model](#merging-the-adapter-with-the-base-model)
   - [Performing Inference](#performing-inference)
8. [Results Parsing](#results-parsing)
9. [Metrics Calculation](#metrics-calculation)

---

## Abstract
Recommender systems (RSs) have become increasingly versatile, finding applications across diverse domains. %As shown by several works, 
Large Language Models (LLMs) significantly contribute to this advancement since the vast amount of knowledge embedded in these models can be easily exploited to provide users with high-quality recommendations.
However, current RSs based on LLMs have room for improvement. As an example, *knowledge injection* techniques can be used to fine-tune LLMs by incorporating additional data, thus improving their performance on downstream tasks. In a recommendation setting, these techniques can be exploited to incorporate further knowledge, which can result in a more accurate representation of the items.
Accordingly, in this paper, we propose a pipeline for knowledge injection specifically designed for RS. First,  we incorporate external knowledge by drawing on three sources: *(a)* knowledge graphs; *(b)* textual descriptions; *(c)* collaborative information about user interactions. Next, we lexicalize the knowledge, and we instruct and fine-tune an LLM, which can then be easily to return a list of recommendations. Extensive experiments on movie, music, and book datasets validate our approach. Moreover, the experiments showed that knowledge injection is particularly needed in domains (*i.e.,* music and books) that are likely to be less covered by the data used to pre-train LLMs, thus leading the way to several future research directions.
---

## Datasets Information
The following datasets are used in this project:

| Dataset       | Users | Items | Ratings | Sparsity  |
|--------------|-------|-------|---------|-----------|
| Last.FM      | 1881  | 2828  | 71,426  | 98.66%    |
| DBbook       | 5660  | 6698  | 129,513 | 99.66%    |
| MovieLens 1M | 6036  | 3081  | 946,120 | 94.91%    |

---

## Apriori Algorithm Parameters
The Apriori algorithm extracts association rules using the following parameters:

| Dataset       | Support  | Confidence | Extracted Rules |
|--------------|----------|------------|-----------------|
| Last.FM      | 0.0015   | 0.002      | 13,391          |
| DBbook       | 0.0003   | 0.001      | 13,245          |
| MovieLens 1M | 0.01     | 0.05       | 62,521          |

---

## LoRA Hyperparameters
LoRA (Low-Rank Adaptation) is used to fine-tune the LLM with the following hyperparameters:

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

---

## Repository Structure
- **DataPreprocessing/**: Preprocessing scripts and knowledge extraction.
- **LLM/**: Scripts for fine-tuning and inference.
- **MetricsCalculation/**: Scripts for evaluating the recommender system.

---

## Data Preprocessing
### Setting Up the Environment
1. Create a virtual environment (Python 3.10.12 recommended):
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows: `env\Scripts\activate`
   ```
2. Install dependencies:
   ```sh
   pip install -r req.txt
   ```

### Generating Training Data
1. **Download Item Descriptions**
   - Obtain files from [this link](https://mega.nz/folder/TsMkQaAB#9vxYcaEZhLcr4005L-bbRg) and place them in dataset folders.

2. **Map DBpedia IDs to Items**
   - Run `dbpedia_quering.py` in the `notebooks/` folder.

3. **Create JSON Training Files**
   - Execute:
     - `Process_text_candidate.ipynb`
     - `Process_graph_candidate.ipynb`
     - `Process_collaborative_candidate.ipynb`
   - Select dataset using the `domain` variable.

4. **Create Training Sets for Ablation Studies**
   - Run `Merge_sources_candidate.ipynb` to merge data sources.

---

## Large Language Model (LLM) Training and Inference
### Creating the Singularity Container
```sh
sudo singularity build llm_cuda121.sif LLM/llm_cuda121.def
```

### Training the Model
```sh
singularity exec --nv llm_cuda121.sif python main_train_task.py
```
- Configure parameters in `config_task.yaml`.

### Merging the Adapter with the Base Model
```sh
singularity exec --nv llm_cuda121.sif python main_merge.py
```
- Configure settings in `config_merge.yaml`.

### Performing Inference
```sh
singularity exec --nv llm_cuda121.sif python main_inference_pipe.py
```
- Adjust `config_inference.yaml`.

---

## Results Parsing
Before calculating metrics, parse the inference results:
1. Use the existing `env` from data preprocessing.
2. Run `Parse_results.ipynb` in `DataPreprocessing/notebooks`.
3. Select the appropriate file and dataset in the first cell.

---

## Metrics Calculation
To evaluate model performance:
1. Create a new environment:
   ```sh
   python -m venv metrics_env
   source metrics_env/bin/activate  # On Windows: `metrics_env\Scripts\activate`
   ```
2. Install dependencies:
   ```sh
   pip install -r MetricsCalculation/Clayrsrequirements.txt
   ```
3. Run metric calculation script:
   ```sh
   python MetricsCalculation/metric_cal.py
   ```
   - Select the dataset in the script.
   - Modify `models_name` to evaluate specific configurations.

---
