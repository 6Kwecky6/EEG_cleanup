EEG_cleanup
===========
This project was meant to check for correlation between acceleration and the indipentent components of EEG signal.

***
Dependencies:
-------------
* Python3 [Go to download page for python](https://www.python.org/downloads/)
* MNE [Read more here](https://mne.tools/stable/index.html)
* Numpy [Read more here](https://numpy.org/)

To install dependencies `pip install numpy mne pyxdf`

***
How to use
----------
To run the code, use 'py signal_preprocessing.py'.
By default, this will save all plots in the folder `EEG_cleanup\data\plots\`, and all csv tables will be saved in `EEG_cleanup\data\`. Make suer that those directories exists.
Plots over time are interractive, and van be explored. Press the help button for instructions.

***
### Parameters
* __-p__/__--plot__ --> This parameter makes it so that we will plot data between each change. 

The rest of the parameters are left mostly untested, and should be used at own risk:
* __-m__/__--method__ --> Followed by a string of which method should be used. Default value is 'fastica'
* __-n__ --> Followed by an integer that deciides how many independent components should be used. Default value will let mne decide dynamically.
