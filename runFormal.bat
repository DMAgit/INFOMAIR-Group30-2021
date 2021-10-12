setx PYTHONPATH %cd%
cd src
python main.py --settings=../formalSettings.json
cmd /k