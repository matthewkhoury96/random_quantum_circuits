# Random Quantum Circuits
Code written in Python 3.5 in order to simulate pseudo-random quantum circuits composed only of Clifford Gates.

## Required Libraries
This code makes use of the following libraries: numpy, scipy, numba, matplotlib. 

## Testing the Code 
To make sure everything is working properly, run following command inside the appropriate directory
```
python test.py 
```
If all of the tests pass, then the everything is ready to use. 

## Code Structure 
In order to run simulations, most modifications should be done in main.py or in utils.py. The details of the simulations should be specified in main.py while plotting functions are located in utils.py. Both files contain examples of how to use the code in order to run your own simulations and create your own plots. Likewise, all simulations should be run using the following command in the approprate directory 
```
python main.py 
```

I would not recommend modifying any of the code written in decompose.py, symplectic.py, test.py, or the properly labelled portions of utils.py unless you have a deep understanding of both the code structure and the mathematical structure of the relevant algorithms. 

## Resources 
Many of the functions in this software are implementations of algorithms described in the following academic papers 
[Improved Simulation of Stabilizer Circuits](https://arxiv.org/pdf/quant-ph/0406196.pdf)
[How to efficiently select an arbitrary Clifford group element](https://arxiv.org/pdf/1406.2170.pdf)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
