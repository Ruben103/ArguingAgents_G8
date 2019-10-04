# Arguing Agents

## Problem Statement

We attempt to augment a standard feed-forward classification neural network with two counter-argument classifiers C_0 and C_1 that serve the purpose of providing a counter arguments against the main classifier M using image classification.

The system architecture is as follows:
![Alt text](./assets/model_arch.png?raw=True "Model Architecture")

## Run Instructions

The program has been implemented in Python3. The program requires the latest versions of the following packages to run:
* Tensorflow (2.0.0)
* Keras (2.3.0)
* Numpy

Command to install the packages:
```pip3 install --no-cache-dir tensorflow keras numpy```

Command to run the program:

```python3 main.py```


### References

*   Trevor JM Bench-Capon and Paul E Dunne.   Argumentation in artificial intelligence.Artificialintelligence, 171(10-15):619–641, 2007.

* Oana Cocarascu and Francesca Toni.   Combining deep learning and argumentative reasoning forthe  analysis  of  social  media  textual  content  using  small  data  sets.Computational  Linguistics,44(4):833–858, 2018.

* Artur SD’Avila Garcez, Dov M Gabbay, and Luis C Lamb. Value-based argumentation frameworksas neural-symbolic learning systems.Journal of Logic and Computation, 15(6):1041–1058, 2005.

* Yann LeCun, Corinna Cortes, and CJ Burges.  Mnist handwritten digit database.AT&T Labs
