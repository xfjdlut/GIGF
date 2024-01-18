# README

## Files Definition

`caches`[create this folder yourself]：Store the processed data files that will be used for model training and testing, where in `goalPlanning` file stores the data generated by the `goal planning` module; `responseGeneration` saves the data generated by `response generation` module.

`config`：Store the pretrain model `GPT2` and `BERT`, which are available for download from `huggingface`.

`data`：Training and test raw data that needs to be processed.

`graph`：The code for processing the data input from the heterogeneous graph and the code for updating the heterogeneous graph.

`logs`[create this folder yourself]：Store the trained model.

`model`：The training code and the model of `goal planning`.

`outputs`[create this folder yourself]：store the generated `goal` and `response`.

`responseGeneration`：The model and the data processing code of `response generation` module. 

`utils`：Data processing code.

`main.py`：The entry file of `goal planning`.

## Environment

`GPU` NVIDIA GeForce RTX 3090

`Python` 3.7

`Pytorch`  1.8.0

```
pip install -r requirments.txt
```

## Running

### Goal Planning 

#### Train

run `main.py`：

```
python main.py --mode train
```
You can change the parameters to train different datasets

#### Test and Generate

```
python main.py --mode test
```

### Response Generation

#### Train

run `run_train.py` in the directory `responseGeneration`

```
cd responseGeneration
python run_train.py
```

#### Generate

run `run_infer.py` in the directory `responseGeneration`

```
cd responseGeneration
python run_infer.py
```

#### Evaluate

run `eval_dialogue.py` in the directory `responseGeneration`

```
cd responseGeneration
python eval_dialogue.py
```
