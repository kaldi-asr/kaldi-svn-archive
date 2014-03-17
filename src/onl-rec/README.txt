OnlineLatticeRecogniser
======================================================

Summary
-------
The code present OnlineLatticeRecogniser,
which wraps online decoder and online 
preprocessing (MFCC, Delta+Delta-Delta or splicing 
and feature transformations-> LDA+MLLT respectively MPE).

The OnlLatticeFasterDecoder just reorganises the functionality 
of LatticeFasterDecoder, so it match the online interface.

The onl-rec-latgen-recogniser-test can be used for decoding 1 wave file.
It demonstrates the OnlineLatticeRecogniser usage.

LICENSE
--------
* The code in this directory is released under the Apache license, Version 2.0: 
  http://www.apache.org/licenses/LICENSE-2.0

Notes
----------
* The Python thin wrapper PyOnlineLatticeRecogniser is in src/pykaldi
* The DEMO in kaldi/egs/vystadial/online_demo directory shows OnlineLatgenRecogniser and PyOnlineLatgenRecogniser with pretrained AM and sample test wave files.
* The extension is actively developed at in https://github.com/UFAL-DSG/pykaldi.
* In March 2014, the PyOnlineLatticeRecogniser recogniser was evaluated on Alex domain.
See graphs evaluating OnlineLatticeRecogniser performance at http://nbviewer.ipython.org/github/oplatek/pykaldi-eval/blob/master/Pykaldi-evaluation.ipynb.
* An example posterior word lattice output for one Czech utterance can be seen at http://oplatek.blogspot.it/2014/02/ipython-demo-pykaldi-decoders-on-short.html
* The OnlineLatticeRecogniser is used in Alex dialogue system (https://github.com/UFAL-DSG/alex).
* This Kaldi fork is developed under Vystadial project: https://sites.google.com/site/filipjurcicek/projects/vystadial .
