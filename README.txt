# README

### Initial Steps
The first thing to be done is to install all of the required packages. To do this, cd into the root directory, and then type:
```bash
pip install -r requirements.txt
```
This code is written in Python 3.6, so please run this code in Python 3.6. 

### Datasets
I used a contraceptive method and a red wine quality dataset from the UCI Machine Learning repository. Links to the two datasets are below:
* Contraceptive dataset: https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
* Wine dataset: http://archive.ics.uci.edu/ml/datasets/Wine+Quality?ref=datanews.io

### Running Code
To run any of this code, simply type `python NAME_OF_FILE.py`.
1. Decision Tree: `python DecisionTree.py`
2. Boosting: `python Boosting.py`
3. k Nearest Neighbors: `python kNN.py`
	* There are two options to run this, running it normally will not output training and testing accuracy curves. If you want this, run
	```v
	python kNN.py --curves
	```
4. Neural Network: `python NeuralNetwork.py`
5. Support Vector Machines: `python SupportVectorMachines.py`
