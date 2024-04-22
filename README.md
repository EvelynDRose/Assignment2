# Assignment2
Assignment2

This code evaluates the different layers of a model (Llama2) on the Standord Alpaca dataset. It then evaluates the models' BLEU, Rogue-L, and BERTScores.

#### Requirements
- Python 	3.11
- Pytorch 	 
- Transformer 	
- datasets
- evaluate
- trl
- peft
- tabulate
- statistics

### Datasets
- Download the Stanford Alpaca dataset at https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#data-release

### Reproducibility
Run the code using python 3

### Task 1
Here are some graphs from the probabilities of generation for the llama2 model with different early exit layers.
!(figure1.png)
!(figure2.png)
!(figure3.png)
### Task 2

### Task 3
+----------+----------+-----------+-------------+
| Layer    |     BLEU |   Rogue-L |   BERTScore |
+==========+==========+===========+=============+
| Layer 8  | 0.503511 |  0.478649 |    0.896649 |
+----------+----------+-----------+-------------+
| Layer 16 | 0.486005 |  0.427759 |    0.900098 |
+----------+----------+-----------+-------------+
| Layer 24 | 0.500413 |  0.528797 |    0.901518 |
+----------+----------+-----------+-------------+
| Layer 32 | 0.54229  |  0.575457 |    0.93543  |
+----------+----------+-----------+-------------+


## References  
- https://huggingface.co/docs/transformers/en/training
- https://www.datacamp.com/tutorial/fine-tuning-llama-2
- https://huggingface.co/docs/peft/main/en/tutorial/peft_model_config
- https://github.com/brevdev/notebooks/blob/main/llama2-finetune-own-data.ipynb
