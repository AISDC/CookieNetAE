# CookieNetAE

This is an open source implementation of the CookieNetAE model, developped by Naoufal Layad and Ryan Coffee, that will be detailed in https://arxiv.org/abs/2105.13967

This repo is mostly for computing performance study, due to the limitted file size allowed by GitHub, we only uploaded a subset of the training/validation data.
In order to make the computing load to be similiar as if full dataset are used, we increased to epoches to 400 so that the number of training steps are similiar.

# Run 
One needs to uncompare all files in dataset folder.
Horovod, instraction here https://horovod.ai, is needed to run the code on multiple GPUs e.g., `horovodrun -np 1 ./main-hvd.py` versus `horovodrun -np 8 ./main-hvd.py`
