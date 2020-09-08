# Identify critical neurons using Mixed Integer Programming

A novel way of computing neuron importance score at fully connected / convolutional layers and using these computed scores to prune non-critical neurons with marginal loss in the accuracy without fine-tuning or retraining.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Install the requirements 

```
pip3 install -r requirements.txt
```

## MIP Solver
We use the Commercial MOSEK. In order to run experiments, a license file mosek.lic is required at location /root/mosek for Ubuntu.
To use another solver, open [sparsify_model.py script](sparsify/sparsify_model.py#L188) and change solver=cp.Mosek to the solver available in the following table (CVXPY is a Python-embedded modeling language for convex optimization problems on top of different solvers).
### Available Solvers for CVXPY
|         	| LP 	| QP 	| SOCP 	| SDP 	| EXP 	| MIP 	|
|---------	|----	|----	|------	|-----	|-----	|-----	|
| CBC     	| X  	|    	|      	|     	|     	| X   	|
| GLPK    	| X  	|    	|      	|     	|     	|     	|
| GLPK_MI 	| X  	|    	|      	|     	|     	| X   	|
| OSQP    	| X  	| X  	|      	|     	|     	|     	|
| CPLEX   	| X  	| X  	| X    	|     	|     	| X   	|
| ECOS    	| X  	| X  	| X    	|     	| X   	|     	|
| ECOS_BB 	| X  	| X  	| X    	|     	| X   	| X   	|
| GUROBI  	| X  	| X  	| X    	|     	|     	| X   	|
| MOSEK   	| X  	| X  	| X    	| X   	| X   	| X  	|
| CVXOPT  	| X  	| X  	| X    	| X   	|     	|     	|
| SCS     	| X  	| X  	| X    	| X   	| X   	|     	|


## Running the experiments

All the experiments reported in the paper are in experiments_notebook.ipynb

## Training Models
    $ python3 train_model.py 
### Arguments
- -sd : parent directory to store logs and models
- -e  : number of training epochs
- -dl : dataset index to be used for training with the following order ['MNIST', 'FashionMNIST', 'KMNIST', 'Caltech256', 'CIFAR10'] 
- -r  : number of training reset to train multiple models with different initializations
- -m  : model index with the following order ['FullyConnectedBaselineModel', 'FullyConnected2Model', 'Lecun Model 98', 'Dense Fully Connected', 'Lenet', 'vgg19']
- -op : optimizer used for training with the following order ['Adam', 'SGD', 'RMSPROP']
- -l  : learning rate index with the following order ['1e-1', '1e-2', '1e-3', '1e-5']
- -bs : batch size used during training
- -dgl: to enable decoupled greedy learning during the training

## Sparsifying Models
    $ python3 run_sparsify.py
### Arguments
starts with same arguments as training to select the right experiment directory with the following extra arguments:
- -tt : pruning threshold (neurons having an importance score below the selected threshold are going to be pruned)
- -sw : \lambda used to control loss on accuracy (more weight will prune less to keep predictive capacity)
- -ft : flag to enable fine tuning after pruning
- -te : number of fine tuning epochs
- -n  : number of data points as input to the MIP
- -mth: a flag when enabled will use mean of layer's importance score as the pruning threshold
- -f  : a flag that forces re-computing the neuron importance score instead of using cached results from previous runs
- -rl : a flag to relax ReLU constraints
- -dgl: to use auxiliary networks trained per layer to compute neuron importance score for large models
- -seq: a flag to run the MIP independently on each class then taking the average
- -bll: a flag to run the MIP on each layer independently starting from the last layer

## Sparsifying every n iterations/epochs
    $ python3 train_sparsify.py
### Arguments
Starts with same arguments as training and sparsify to select the right experiment directory with the following extra arguments
  - -trst : a flag to run sparsify every n iterations, if disabled will run every n epochs
  - -ent  : an integer for n between epochs/iterations to apply sparsification
  - -incr : a flag to enable incremental training of computed sub-network  

## Robustness to different batches Experiments
    $ python3  verify_selected_data.py
### Arguments
Starts with same arguments as sparsifying models to plot the pruning percentage and the accuracy changes when the batch of images fed to the MIP changes.

## Different Lambdas Experiments
    $ python3 plot_different_lambdas
### Arguments
Starts with same arguments as sparsifying models to plot the pruning percentage and the accuracy changes when the value of the \lambda (-sw) changes.

## Average runs on different classes robustness
    $ python3 batch_data_experiments.py
### Arguments
Starts with same arguments as run_sparsify.py with the following extra arguments:
  - -nex: an integer denoting the number of experiments conducted
  - -bbm: a flag when enabled, we sample a balanced set of images per class, otherwise a random number of images per class is used
  - -ppexp: a flag when enabled the MIP runs independently per class and the neuron importance score becomes the average of multiple runs

## References
```
@article{elaraby2020identifying,
  title={Identifying Critical Neurons in ANN Architectures using Mixed Integer Programming},
  author={ElAraby, Mostafa and Wolf, Guy and Carvalho, Margarida},
  journal={arXiv preprint arXiv:2002.07259},
  year={2020}
}
@article{mosek2010mosek,
  title={The MOSEK optimization software},
  author={Mosek, APS},
  journal={Online at http://www. mosek. com},
  volume={54},
  number={2-1},
  pages={5},
  year={2010}
}
@article{cvxpy,
  author  = {Steven Diamond and Stephen Boyd},
  title   = {{CVXPY}: A {P}ython-Embedded Modeling Language for Convex Optimization},
  journal = {Journal of Machine Learning Research},
  year    = {2016},
  volume  = {17},
  number  = {83},
  pages   = {1--5},
}
@article{cvxpy_rewriting,
  author  = {Akshay Agrawal and Robin Verschueren and Steven Diamond and Stephen Boyd},
  title   = {A Rewriting System for Convex Optimization Problems},
  journal = {Journal of Control and Decision},
  year    = {2018},
  volume  = {5},
  number  = {1},
  pages   = {42--60},
}
```

## License
MIT license
