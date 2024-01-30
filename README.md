# Tree Variational Autoencoders
This is the code for the NeurIPS 2023 Publication:

For running TreeVAE (or its variational baselines):
1. Create a new environment with the treevae.yml or minimal_requirements.txt file
2. Select the dataset you want use by changing the default of config_name in the main.py parser (or baselines/lvae/main_lvae.py respectively)
3. Potentially adapt default configuration in the config of the selected dataset
4. For Weights & Biases support, set project & entity in train/train.py (or baselines/lvae/train_lvae.py respectively) and change value of "wandb_logging" to 'online' in the selected config file
5. Run main.py (or baselines/lvae/main_lvae.py respectively)

This PyTorch repository was thoroughly debugged and tested, however, please note that the experiments of the submission were performed using the repository with the Tensorflow code.