





# Install dependencies

```bash
pip install -r requirements.txt
```



# To train the models



**To train the CNN model, simply running the follow script.**

```bash
python scripts/train_cnn.py
```

*If you run this for the first time, it will around 30mn (not include training) for downloading resources and preprocessing.*



To customize the features, edit feature array in each training script, e.g.,

```python
# features containing only CamemBert embedding
features_args =['CamemBertEmbeddingFeature'] 

# features containing only FastText Embedding and Word Length
features_args =['FastTextEmbeddingFeature', 'WordLengthFeature'] 
```

`train_and_evaluate_n_times(features_args, n=1)` # n=1 mean the model one time, n=5 train 5 times.

`All the model checkpoints and report will be saved to the folder models/FR/*`



**Here is the list of all features**

* FastTextEmbeddingFeature
* CamemBertEmbeddingFeature
* WordLengthFeature
* WordSyllableFeature
* VowelCountFeature
* TFIDFFeature
* WordRankFeature
* LangGenFrequencyFeature
* ClearFrequencyFeature





**Train CatBoost Model**

```bash
python scripts/train_catboost.py
```



**Train CatBoost Model**

```bash
python scripts/train_catboost.py
```

