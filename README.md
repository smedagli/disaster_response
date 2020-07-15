# Disaster response classifier
Udacity project:
categorizing tweets in case of disaster and catastrophic events
###### Author: Stefano Medagli
###### date: 14.06.2020
###### ver: 0.1
## prerequisites
create a conda environment using the file in `environment/pkg.txt`

```bash
conda create --name disaster --file disaster_response/environment/pkg.txt
```
#### @TODO:
    fix GridSearchCV in function `build_model` for TextExtractor

## folder structure
```bash
|   .gitignore
|   paths.py
|   README.md
|   __init__.py
|       
+---data
|   |   categories.csv
|   |   messages.csv
|   |   messages.db
|   |   process_data.py
|   |   process_data_script.py
|   |   __init__.py
|           
+---envitonment
|       pkg.txt
|       tree.txt
|       
+---language
|   |   custom_extractor.py
|   |   nltk_functions.py
|   |   __init__.py
|   |   
|           
+---models
|       (model.pkl)
|       metrics.py
|       train_classifier.py
|       train_classifier_script.py
|       __init__.py
|       
```
## pre-processing data
To preprocessing of the data.
Run the script as
```bash
python data/process_data.py <messages_csv_path> <categories_csv_path> <database_db_path>
```
will import messages and categories from the 2 .csv files,
merge them and save into the .db file

Alternatively run `data/preprocess_data_script.py`.
For more information run the script with `-h` option.

## build/train model
To train the model, run
```bash
python models/train_classifier.py <.db file> <.pkl file>
```
It will load the messages and labels from the .db file,
train a model for text prediction and save it to the .pkl file.

The model is defined inside the function `build_model()` 

Alternatively run `models/train_classifier_script.py`
For more information run the script with `-h` option.

### components
* *paths.py*:
the module defines the default paths of the project

#### app
* *run.py*:
run this script to get the web app for classification (will use 'localhost')
#### data
* *process_data.py*:
reads from the .csv files for messages and categories and writes a .db file
* *process_data_script.py*:
same as process_data.py but with a different way to parse input from command line
#### language
* *custom_extractor.py*:
defines custom transformer classes for text analysis
* *nltk_functions.py*:
contains common/quick functions for natural language processing 
#### models
* *metrics.py*:
contains methods to compute and print model's performance
* *train_classifier.py*:
defines the model and trains it. Then saves the model as .pkl file
* *train_classifier_script.py*:
same as train_classifier.py but with a different way to parse input from command line  

