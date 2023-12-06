# Model Shapley: Equitable Model Valuation with Black-box Access [NeurIPS'23]

Implementation for our work "Model Shapley: Equitable Model Valuation with Black-box Access" in the Thirty-seventh Conference on Neural Information Processing Systems 2023 (**NeurIPS'23**):
**Xinyi Xu**, _Thanh Lam_, _Chuan Sheng Foo_, _Bryan Kian Hsiang Low_. | [Paper](https://openreview.net/pdf/a7a54583b4f1c783becf9d430f57ee2e68181b16.pdf) | [Poster](https://nips.cc/media/PosterPDFs/NeurIPS%202023/71312.png?t=1701417342.575461) | [Presentation](https://nips.cc/media/neurips-2023/Slides/71312.pdf) | 


## Usage

We provide the following instructions on using this implementation. The python environment is managed through anaconda, and the `environment.yml` file provides the used packages. 

There are several key parts in the implementation, outlined here: 
   - Part I: training and saving ML models (weights) on specified datasets and hyperparameters;
   - Part II: loading the saved ML models, obtaining their Dirichlet abstractions and computing/estimating Model Shapley values;
   - Part III: (optional) perform regression learning on Model Shapley values (MSVs), using Gaussian process regression;
   - Part IV: others, including the networking plot in Fig. 11 (in our full paper).

### Environment
Tested OS platform: Ubuntu 20.04 with Nvidia driver version 515.43 and CUDA Version: 11.7.

`conda env create -f environment.yml`

### Part I (Training & saving models)
> Example: training models on MNIST with varying model types for $10$ models each type:

`python run_mnist.py -N 10 -t models`

> Example: training models on CIFAR-10 with varying dataset sizes for $5$ models each dataset size:

`python run_cifar10.py -N 5 -t datasets`

Executing the above command will create a directory `save_models` under which the weights of the trained Pytorch models are saved for loading later.

### Part II (Loading models, obtaining Dirichlet abstractions & computing Model Shapley values)

> Example: obtaining the Dirichlet abstractions, using by-class partition, and the MSVs for **completed** model training of MNIST with $10$ models under different model types.

`python get_MSVs.py -d MNIST -iN 10 --by_class -t models`

> Example: obtaining the Dirichlet abstractions, *not* using by-class partition, and the MSVs for **completed** model training of MNIST with $10$ models under different model types.

`python get_MSVs.py -d MNIST -iN 10 --not_by_class -t models`

Executing the above command will create a directory `MNIST_results` (if dataset is MNIST) under which the results are saved (in the corresonding subdirectories).

### Part III (Learning Model Shapley values via regression)

> Example: performing learning of MSVs on saved Dirichlet abstractions (saved alphas) for MNIST and using RBF kernel for the ExactGP inference.

`python learn_MSV.py -d MNIST --not_l1`

> Example: performing learning of MSVs on saved Dirichlet abstractions (saved alphas) for MNIST and using $\ell_1$ distance for the ExactGP inference.

`python learn_MSV.py -d MNIST --l1`

Note: need to provide the right directory storing the saved alphas in the code.

Executing the above command will create a directory `learn_MNIST_l2` (if dataset is MNIST) under which the results are saved (in the corresonding subdirectories).

### Part IV (Others) [Optional]

#### Pruning in an ensemble learner.

Check out `deeplearning_ensemble.py` and `deeplearning_ensemble_SV_multiple.py`.

> Example: performing deepensemble pruning on the MNIST dataset, with $20$ base learners. The ensemble (voting classifer) is trained for $20$ epochs. The entire experiment is repeated for $3$ times.

`python deeplearning_ensemble_SV_multiple.py -d MNIST -N 20 -E 20 -M 3`

Executing the above command will create a diretory `DeepEnsemblePruning_results` and save the results (and logs and the checkpoints of the ensemble learner) therein. Base learners (e.g., "LeNet5" for MNIST) will be trained and then used to construct an ensemble learner (i.e.., the `VotingClassifier` class from "torchensemble"), and the pruning based on the (estimated) MSVs of the base learners is performed and the results plotted. 

#### Plot the clustering of Dirichet abstractions, using Chernoff distance

Need to first have computed multiple models (e.g., $50$ of MLPs, CNNs, LRs each for MNIST for a total of $150$ points in the cluster.) Then use `draw_network.py` by loading the correct distances from the right directories.


## Citing
If you have found our work to be useful in your research, please consider citing it:
```
@inproceedings{Xu2023,
   author = {Xu, Xinyi and Lam, Thanh and Foo, Chuan Sheng and Low, Bryan Kian Hsiang},
   booktitle = {Advances in Neural Information Processing Systems},
   title = {Model {S}hapley: Equitable Model Valuation with Black-box Access},
   year = {2023}
}
```
