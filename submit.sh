#!/usr/bin/env bash
echo "############################## start test  use 8 gpus #####################################"
./tools/dist_test.sh configs/ReDet/SAR_model_config.py work_dirs/SAR_model_config/latest.pth 8 --out work_dirs/SAR_model_config/results.pkl
echo "###################### parse result ##############################"
python ./tools/parse_results.py
echo "##################### process final result ##################"
python process.py
zip -r final_result.zip ./SAR_txt_result/*
