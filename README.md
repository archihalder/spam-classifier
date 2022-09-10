# Spam Classifier

## Description
Classify whether the messages are spam or ham (legitimate).

---

## Dataset
- This dataset is collected from UCI Machine Learning Repository. Link to the website is given <a href="https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection">here</a>
- It contains 5574 messages out of which around 86% of them are legitimate messages and about 13% of them are spam. 
- To get more information about the dataset, click [here](data/README.md)

---
## How to use this repo

1. Clone the project to you local machine

```bash
git clone git@github.com:archihalder/spam-classifier.git
```

2. Enter the directory

```bash
cd spam-classifier
```

3. Create a virtual environment in your current directory
```bash
pip install virtualenv
virtualenv spam-env
source spam-env/bin/activate
```

4. Get the required modules to run

```bash
pip install -r requirements.txt
```

5. Install the complete `nltk` module. Write the following in Python Shell
```python
import nltk
nltk.download('all')
```

6. Run the file

```bash
python3 src/spam-classifier.py
```
---
## Results and Observations

- Model used - Naive Bayes Classifier
- Accuracy achieved - 98.4%