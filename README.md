# Autogluon2onnx
This repository works on converting autogluon model and feature generator to ONNX file

#Introduction 
The converter is based on the generated Autogluon model files. It only supports LightGBM bagged_model so far.

# Usage 
## make sure the models folder is at the same directory with the script file.
python bagged_ensemble_onnx.py --model_dir 
#e.g. python bagged_ensemble_onnx.py --model_dir C:\Users\Dell\Desktop\nok_autogluon\AutogluonModels\ag-20230203_070455\models\LightGBMLarge
