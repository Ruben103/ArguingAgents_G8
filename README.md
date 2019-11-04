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
* OpenCV
* URLLIB

The dataset can be downloaded from the following link:
![Cats-vs-Dogs](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip)

Download the dataset (zip file) and extract to ```./data/PetImages```

Command to install the packages:
```pip3 install --no-cache-dir tensorflow keras numpy opencv-python```

Command to run the ensemble of main and counter argument classifiers:

```python3 main.py```

Command to run the baseline main classifier:
```python3 just_main.py```


### References

*   Trevor JM Bench-Capon and Paul E Dunne.   Argumentation in artificial intelligence.Artificialintelligence, 171(10-15):619–641, 2007.

* Oana Cocarascu and Francesca Toni.   Combining deep learning and argumentative reasoning forthe  analysis  of  social  media  textual  content  using  small  data  sets.Computational  Linguistics,44(4):833–858, 2018.

* Artur SD’Avila Garcez, Dov M Gabbay, and Luis C Lamb. Value-based argumentation frameworksas neural-symbolic learning systems.Journal of Logic and Computation, 15(6):1041–1058, 2005.

* Yann LeCun, Corinna Cortes, and CJ Burges.  Mnist handwritten digit database.AT&T Labs

* Elson, J., Douceur, J. J., Howell, J., & Saul, J. (2007). Asirra: a CAPTCHA that exploits interest-aligned manual image categorization.
