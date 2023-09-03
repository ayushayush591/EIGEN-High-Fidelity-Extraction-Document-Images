# key-value-extraction-from-spear
---------------------------------------------------------------------------------------------------------------------------------------------------------------
###Instruction For Training<br>
Create a new virtual environment, navigate to this directory and run the following command:
1. git clone main branch.<br>
2. Download CORDS receipt dataset in current directory "https://drive.google.com/drive/folders/1mKrsYBW7xXzfxNLSYwQ02bHayqVfe-94?usp=sharing".
3.  ```pip install -r requirements.txt``` for installing all the dependency.
4.  ```git clone https://github.com/iitb-research-code/spear4HighFidelity.git```to get all the required files to run spear and CAGE.<br>
5. Then change labeling function as per your need, Ex- adding or removing labeling function and make appropriate changes.(optional).<br>
6. Run labeling_function file ```python main.py```
7. Your pickle file which was required for training and trained Model files will get store in Paths folder.<br>
---------------------------------------------------------------------------------------------------------------------------------------------------------------
###Files information<br>
1. Cage_cords.ipynb is the file which contains code for running CAGE model on Cords dataset.<br>
2. NH_cage.ipynb is the file which contains code for running CAGE model on NH dataset.<br>
3. Paths directory contain all the pickle files which is needed for training.<br>
4. cords_demo.ipynb is the file which contains code for running inference on CORDS data from the stored model.<br>
5. nh_demo.ipynb is the file which contains code for running inference on NH data from the stored model.<br>
