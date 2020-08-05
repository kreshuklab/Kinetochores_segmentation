## File description

The files in this directory are the modified versions of the files from pytorch-3dunet implementation by Adrian.
losses module contains the new GaussianDiceLoss function implemented for the super resolution like loss in separating foreground and background.
Smoothing module contains the Smoothing gaussian function - pytorch version
trainer routine is similar with minor comments
train_dice_egfp.yaml contains the yaml to be passed to train the network.
test_dice_egfp.yaml contains the yaml for testing and get the predictions.

Add these files in pytorch-3dunet, run setup.py install and run the training routine.

