# Integrating Heterogeneous Knowledge for Enhanced Recommendation with Large Language Models

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

The integration of Large Language Models (LLMs) into recommender systems has introduced a new paradigm wherein models leverage their extensive pre-trained knowledge to generate personalized suggestions. A prevailing assumption is that an LLM's inherent knowledge is a sufficient foundation for high-quality recommendations across diverse domains. This paper challenges that assumption, positing that in specialized domains, recommendation efficacy is fundamentally limited by the textual nature of an LLM's knowledge, which often excludes critical non-textual signals such as collaborative patterns, structured attributes, and multimodal features.

To address this limitation, we propose REKALM, a novel framework for enhancing LLM-based recommenders through targeted knowledge integration. Central to our approach is \textbf{knowledge lexicalization}, a process that translates heterogeneous data sources, including collaborative, factual, and multimodal knowledge, into a unified natural language format. This lexicalized corpus is then used in a \textbf{knowledge-aware instruction-tuning} phase to explicitly align the LLM's internal representations with domain-specific information. We conduct extensive experiments across four distinct domains, movies, books, music, and board games—to validate our framework. Our findings provide empirical evidence that while an LLM's inherent knowledge may suffice for universally familiar domains like movies, recommendation quality in more specialized areas is significantly improved through our knowledge integration methodology. The proposed approach outperforms strong state-of-the-art baselines, demonstrating that explicitly augmenting LLMs with lexicalized, domain-specific knowledge is a critical and effective strategy for advancing the next generation of recommender systems.

---

## Datasets Information
The following datasets are used in this project:

| Dataset | Users | Items | Ratings | Sparsity | Images | Audio | Video |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MovieLens 1M | 6,036 | 3,081 | 946,120 | 94.91% | 3,081 | 3,105 | 3,105 |
| DBbook | 5,660 | 6,698 | 129,513 | 99.66% | 4,276 | - | - |
| Last.FM | 1,881 | 2,828 | 71,426 | 98.66% | 2,820 | 2,820 | 2,742 |
| Boardgames Geek      | 168,733     | 21,772 | 6,279,384     | 99.82% | 21,616 | - | - |
---

## Apriori Algorithm Parameters
The Apriori algorithm extracts association rules using the following parameters:

| Dataset | Support | Confidence | Extracted Rules |
| :--- | :---: | :---: | :---: |
| MovieLens 1M | 0.01 | 0.05 | 62,521 |
| DBbook | 0.0003 | 0.001 | 13,245 |
| Last.FM | 0.0015 | 0.002 | 13,391 |
| Boardgamegeek     | 0.001 | 0.002 | 15,328 |

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
