# comp-491-bitirme
Natural Language Guided Deep Reinforcement Learning

Senior project, replicating: https://arxiv.org/abs/1704.05539

First we create a dataset from videos of us playing Montezuma's Revenge, relevant material is in createdataset

Then we convert the dataset into a form that can be processed by the bimodal classifiers we implemented. That code is available in the bimodal classifier folder

Lastly, using the bimodal classifier we pretrained, we train a natural language guided deep RL agent on Montezuma's Revenge. Relevant code is in Montezuma's Revenge.

The code is not cleaned and if you need help understanding parts of it you can mail us at
{etekalp16, okolukisa16}@ku.edu.tr
