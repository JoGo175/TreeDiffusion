
# Enhancing TreeVAE: Advancements in Modeling Hierarchical Latent Spaces for Improved Variational Autoencoders

**Master Thesis**

Author: Jorge Fernando da Silva Gonçalves.

Submission Date: March 15, 2024

Supervisors Prof. Dr. Julia Vogt, Dr. Markus Kalisch

Advisors: Moritz Vandenhirtz, Laura Manduchi






This repository contains the implementation of **CNN-TreeVAE** and **Diffuse-TreeVAE**. 



##  Ablation Study and CNN-TreeVAE training

`cluster_run.sh` replicates the ablation study results for the selected dataset. To change the selected dataset, you need to change the "dataset" variable within the script. 

Note: For CIFAR-10, use 0.01 instead of 0.0 in the kl_start loop


`cluster_run_simple` can instead be used to train one CNN-TreeVAE mdoel at a time.


## Diffuse-TreeVAE training

To train the Diffuse-TreeVAE models, make sure to have the correct model configurations in the respective configs file. 
Furthermore, you need to add the folder name of the pre-trained CNN-TreeVAE models on which the Diffusion models are conditioned. 

To train multiple Diffusion models, use the `train_ddpm_{dataset}.sh` for the respective dataset. 
If only one model is trained, you can use `train_ddpm.sh` instead.


To retrieve the reconstructions or samples from the diffusion model, further adjust the `test_ddpm_{dataset}.sh` script with the corresponding paths to the trained CNN-TreeVAE and trained DDPM models. 




## Remarks on the repo structure

This repo closely follow the public repo of the original implementation (https://github.com/lauramanduchi/treevae). 


The folders are structured as follows: 
- baselines: contain LadderVAE which was used as baseline in the Tree Variational Autoencoder paper
- configs: contains config files with the config parameters for each dataset
- data: data gets downloaded into this folder and is used for the models
- FID: implements the code to compute the FID scores
- images: contains some images for visualization 
- models: contains the python files that build the TreeVAE model and the Diffusion model 
- notebooks: contains various jupyter notebooks that were used for data analysis and to create images
- results: folder for the Diffuse-VAE training and evaluation
- scripts: contains the scripts that were used to run the models on the ETH LEOMED cluster. 
- train: contains python files to train the TreeVAE
- utils: contains several python files with utility functions
- main.py: runs the TreeVAE training and evalutation
- train_ddpm.py: trains the DDPM based on a TreeVAE model
- test_ddpm.py: generates the reconstructions/samples for the DDPM 

