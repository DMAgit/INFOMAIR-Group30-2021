setx PYTHONPATH %cd%
cd src
python main.py --settings=../informalSettings.json
cmd /k