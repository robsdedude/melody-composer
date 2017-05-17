This program is part of my bachelor thesis which can be found on
[arXiv](https://arxiv.org/pdf/1612.09212.pdf). It's a machine learning algorithm
that generates monophonic melodies similar to those of a phrase annotated
data-set. In- and output are midi files (i.e. machine readable music).

Setup
=====
Use virtualenv
--------------

    virtualenv venv
    source venv/bin/activate
    export LC_ALL=C
    pip install -r requirements.txt

Troubleshooting
---------------
On Ubuntu based systems it can happen, that there are header files missing. Do:
`sudo apt-get install liblapack-dev libblas-dev`
It can also happen, that you need to install a fortran compiler. Do:
`sudo install gfortran`
After fixing these things re-run `pip install -r requirements.txt` each time.

Trainings Data
-------------
Download the [MTC-FS data set](http://www.liederenbank.nl/mtc/collections.php)
and unzip it into the folder `MTC-FS-1.0`

Then run the preporcessing

    mkdir -p MTC-FS-1.0/good_midi  # create folder to put pre-processed midi files in
    source venv/bin/activate       # activate virtualenv
    python preprocess.py           # run preprocessing
    deactivate                     # deactivate virtualenv


Run the program
===============
Start the virtual environment with
`source venv/bin/activate`
To general start the program go to this directory and run
`python main.py`
More information: `python2 main.py --help`


Other things
============
To see how the clustering is working have a look at main() in contour.py

