

# Intro 
These are the instruction to re-run the PLAiD work.

## PLAiD

 1. Train the flat walking controller
 
 ```
 python3 trainModel.py --config=settings/terrainRLImitate/CACLA/A_CACLA_DeepNN.json
 ```
 Each method started out from this one trained model. This helps reduce variance in the method comparison.
 
  1. Train the Incline controller
 
 ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/Inclines_TF_From_Flat.json settings/hyperParamTuning/agility_PLAiD.json   5 5
 ```
   You might need to copy the model from the previous step into 5 sub folders for this to run properly over the 5 simulations

   1.  Create a network with new terrain features.
       1. Copy the trained incline controllers into folders for the model with a CNN
       1. You Will need to do this for the number of simulation samples being done.
       1. 
```
python3 combineNetworkModels.py settings/terrainRLImitate/CACLA/Steps2_zero_merge.json
```

   1. Distill the controllers together.
 ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/Distillation_Flat_And_Incline.json settings/hyperParamTuning/agility_PLAiD_Distill_Flat_And_Inclines.json 5 5
 ```
    	
   1. TL Train the steps controller
 ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/PLAiD_Steps.json.json settings/hyperParamTuning/agility_PLAiD_Steps.json 5 5
 ```
 
   1. Distill the controllers together.
 ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/Distillation_Flat_And_Incline_Steps.json settings/hyperParamTuning/agility_PLAiD_Distill_Flat_And_Inclines_Steps.json 5 5
 ```
 
   1. TL Train the slopes controller
 ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/PLAiD_Slopes.json settings/hyperParamTuning/agility_PLAiD_Slopes.json 5 5
 ```
 
   1. Distill the controllers together.
 ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/Distillation_Flat_And_Incline_Steps_Slopes.json settings/hyperParamTuning/agility_PLAiD_Distill_Flat_And_Inclines_Steps_Slopes.json 5 5
 ```
 
   1. TL Train the gaps controller
 ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/PLAiD_Gaps.json settings/hyperParamTuning/agility_PLAiD_Gaps.json 5 5
 ```
 
   1. Distill the controllers together.
 ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/Distillation_Flat_And_Incline_Steps_Slopes_Gaps.json settings/hyperParamTuning/agility_TL_Only_Distillation.json 5 5
 ```
 
 
## TL-Only

The instructions in these steps are similar to the PLAiD ones. Different settings files are used here so that the proper models are copied and load and not mixed with PLAiD.
 
  1. Use the same initial walking controller from PLAiD
  
  1. Use the same method as PLAID because so far they are identical
  
  1.  Use the same created network from PLAiD
      
    	
   1. TL Train the steps controller
 ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/TL_Only_Steps_TL_From_Inlines.json settings/hyperParamTuning/agility_TL_Only_Inline_To_Steps.json 5 5
 ```
 
   1. TL Train the slopes controller
 ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/TL_Only_Slopes_TL_From_Steps.json settings/hyperParamTuning/TL_Only_Steps_TL_From_Inlines.json 5 5
 ```
 
   1. TL Train the gaps controller
 ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/TL_Only_Gaps_TL_From_Slopes.json settings/hyperParamTuning/agility_TL_Only_Slopes_To_Gaps.json 5 5
 ```
 
   1. Ontional step to distill the TL-Only controllers together.
 ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/TL_Only_Distillation.json settings/hyperParamTuning/agility_TL_Only_Distillation.json 5 5
 ```
 
## MultiTaskter

 
  1. Use the same initial walking controller from PLAiD
  
  1.  Create a network with terrain features for the Multitasker
  
      1. Copy the trained incline controllers into folders for the model with a CNN
       1. You Will need to do this for the number of simulation samples being done.
       1. 
```
python3 combineNetworkModels.py settings/terrainRLImitate/CACLA/Steps2_zero_merge.json
```
  
  1. Add the incline task to the set of task to learn and train.
  
   ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/MultiTasking_Flat_Incline.json settings/hyperParamTuning/activation_MultiTasker.json 5 5
 ```
    	
  1. Add the steps task to the set of task to learn and train.
 ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/MultiTasking_Flat_Incline_Steps.json settings/hyperParamTuning/activation_MultiTasker_Flat_Incline_Steps.json 5 5
 ```
 
  1. Add the slopes task to the set of task to learn and train.
 ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/MultiTasking_Flat_Incline_Steps_Slopes.json settings/hyperParamTuning/activation_MultiTasker_Flat_Incline_Steps_Slopes.json 5 5
 ```
 
  1. Add the slopes task to the set of task to learn and train.
 ```
 python3 trainMetaModel.py settings/terrainRLImitate/CACLA/MultiTasking_Flat_Incline_Steps_Slopes_Gaps.json settings/hyperParamTuning/activation_MultiTasker_Flat_Incline_Steps_Slopes_Gaps.json 5 5
 ```
 
### Notes:

  1. These simulation may try to send emails when they are complete, containing the simulation data. You may need to update the hyperSettings files to send them to your email address, not mine...
  
  1. You can combine the data from multiple simulations into one plot using 
  ```
  python3 tools/plot_meta_simulation.py <settings_file_name> <path_to_data> <path_to_data> <path_to_data> ...
  ```
  