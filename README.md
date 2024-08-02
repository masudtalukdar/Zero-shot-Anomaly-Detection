# AnomalyCLIP 
> [**ICLR 24**] [**AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection**](https://arxiv.org/pdf/2310.18961.pdf)
>

## Introduction 
In this project, I implemented and evaluated AnomalyCLIP, a novel object-agnostic prompt learning method for zero-shot anomaly detection. The primary goal was to reproduce the results presented in the research paper in the Table 1 using the VisA dataset. Due to computational constraints, I focused on a subset of the dataset, specifically four classes out of the original twelve



## How to Run
### Prepare your dataset
Download the dataset below:

* Industrial Domain:
 [Visa](https://github.com/amazon-science/spot-diff)


### Generate the dataset JSON
```bash
cd generate_dataset_json
python visa.py
```
Select the corresponding script and run it (we provide all scripts for datasets that AnomalyCLIP reported). The generated JSON stores all the information that AnomalyCLIP needs.
