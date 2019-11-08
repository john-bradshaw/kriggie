# kriggie

A _playground_ for me to for GPs (Gaussian Processes [1]) in PyTorch! In fact we don't _actually_ even have a plain 
GP implementation in here currently -- just an SVGP [2].

PyTorch 1.1, Python 3.6


## If you're looking for a fuller featured GP library:

* [GPflow](https://github.com/GPflow/GPflow) Python/TensorFlow. 
A lot of the design of this library is inspired by this project. I have found this a great library to use!
* [GPML](http://www.gaussianprocess.org/gpml/code/matlab/doc/index.html) Matlab
* [GPy](https://github.com/SheffieldML/GPy) Python/Numpy
* [GPytorch](https://github.com/cornellius-gp/gpytorch) Python/PyTorch.
* [pygp](https://github.com/mwhoffman/pygp) Python/Numpy
* [Pyro](https://pyro.ai/examples/gp.html) Python/PyTorch
* [Stheno.jl](https://github.com/willtebbutt/Stheno.jl) Julia

Among many others!


## References
1. Williams, C.K. and Rasmussen, C.E., 2006. Gaussian processes for machine learning Cambridge, MA: MIT press.
2. Hensman, J., Matthews, A. and Ghahramani, Z., 2015. Scalable variational Gaussian process classification. 
