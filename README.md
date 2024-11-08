﻿# symmetry-in-variational-qml
This project follows the paper 'Exploiting symmetry in variational quantum machine learning', arXiv:2205.06217v1.
We used tensorflow and pennylane to implement the toy models suggest by the above paper. Two models are explored so far, 
1) tic-tac-toe game (D_4-symmetry)
2) autonomus vehicle scenerios ($\mathbb{Z}_4$-symmetry).
Both models are studied under the context of a 3x3 grid system. 

For each model. two approaches of utilizing the symmetry of the model are investigated.
The first method, we follow the procedures outline by the above paper, using the invariant re-uploading model and the loss function suggested by the paper. Just out of curiosity, the second method jsut symmetrize the input data and then we train the symmetrized data using a trivial neural network. 

We hope to show by considering symmetry of the model, we can increase the efficiency of training and protect the model from overfitting. 
