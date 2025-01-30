# Empowering Recommender Systems based on Large Language Models through Knowledge Injection Techniques
![Alt text](img/Architecture.png)

## LoRA Hyperparameters
| **Parameter**               | **Value**         |
|-----------------------------|-------------------|
| r                           | $64$              |
| alpha                       | $128$             |
| target                      | all linear layers |
| sequence length             | $2048$            |
| learning rate               | $0.0001$          |
| training epochs             | $10$              |
| weight decay                | $0.0001$          |
| max grad norm               | $1.0$             |
| per device train batch size | $4$               |
| optim                       | adamw torch       |



Following the architecture phases, the repo is structured in 3 main part (and so folders):
- Data Preprocessing that include both the data preprocessing of the user and the knowledge extraction phase.
- LLM repo that include both code for perform fine-tuning and inference on the model
- Output refinement and metric calculation

## Data Preprocessing
first