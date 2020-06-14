# Disaster response classifier
Udacity project: categorizing tweets in case of disaster and catastrophic events
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
|       train_classifier.py
|       train_classifier_script.py
|       
```
## components
### data
* *process_data.py*: reads from the .csv files for messages and categories and writes a .db file
* *process_data_script.py*: same as process_data.py but with a different way to parse input from command line
### language
* *custom_extractor.py*: defines custom transformer classes for text analysis
* *nltk_functions.py*: contains common/quick functions for natural language processing 
### models
* *train_classifier.py*: defines the model and trains it. Then saves the model as .pkl file
* *train_classifier_script.py*: same as train_classifier.py but with a different way to parse input from command line  

