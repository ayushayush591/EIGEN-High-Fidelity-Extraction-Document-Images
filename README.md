# key-value-extraction-from-spear
---------------------------------------------------------------------------------------------------------------------------------------------------------------
###Instruction For Training<br>
Create a new virtual environment, navigate to this directory and run the following command:
1.  git clone dev branch.<br>
2.  ```pip install -r requirements.txt``` for installing all the dependency.
2.  Firstly run ```pip install decile-spear``` to get all the required files to run spear and CAGE.<br>
3.  Make the following changes in Spear library:
    * In file "spear/labeling/analysis/core.py" change line 335 or 336 just find similar line and replace to:
        ```
        confusion_matrix(Y, self.L[:, i], labels=labels)[1:, 1:] for i in range(m)
        ```
    * In file "spear/utils/utils_jl.py in line 54 replace by:    
         ```
         return (probs_p * log(probs_p / (probs_q+1e-15))).mean(dim=0).sum()
         ```
4. Then change labeling function as per your need, Ex- adding or removing labeling function and make appropriate changes then run labeling_function.py file ```python labeling_function.py```.<br>
5. Your pickle file which was required for training and trained Model files will get store in NH_paths folder.<br>
---------------------------------------------------------------------------------------------------------------------------------------------------------------
###For Demo

1. Run doctr_.py which will store the result of doctr_model in NH_paths.<br>
2. then run demo.py to get the result from our model in form of image with bounding boxes labeled.<br>
