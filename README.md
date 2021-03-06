# INFOMAIR-Group30-2021

This is a repository for the group assignments given in the Methods in AI Research (INFOMAIR) course at Utrecht University. 

Group 30 members:
* Dimitar Angelov
* Rienk Fidder 
* Ignacio Montes Álvarez 
* Thimo Poortvliet 

## Starting guide
To run the program, first install the dependencies in requirements.txt using pip:\
`$ pip install -r requirements.txt`

Next, the program can be started from main.py:\
`$ python main.py`

### Settings
The program can be configured using a settings file. This file should be a json.
The default settings file is already present in the repository.
When starting the program, it will ask for a path to the settings file. To use the default settings, answer with an empty string.
Otherwise, give a path to the file relative to the location on which the program is run.

Alternatively, the `--settings` flag can be used when starting from main to specify a settings path.
For example, to run with the settings file in the current folder, run:\
`$ python main.py --setings mySettings.json`

The current available settings are:

* chooseClassifier: false to use the best classifier according to our selection (sgd) and true to see the list of possible pretrained classifiers to use (including the baselines)
* levenshteinDistance: the threshold to use in preference extraction (food, area, price and additional requests as phone or postal code)
* informal: switch between formal and informal system utterances
* addDelay: amount of seconds of delay between answers
* useCaps: switch between lower and uppercase system responses
* tts: enable or disable the text-to-speech of the system utterances

### Development mode
The models used in the program are pre-trained and saved in the repository, but it is also possible to train them yourself and see the training results. 
To do this, the program needs to be ran with the "development" flag as such:\
`$ python main.py --development`
