Creating a multihead model for pytorch-3dunet

Files to be changed:

Data Loading - hdf5.py: utils.py

hdf5.py contains the dataloader processing and loader routine.
In the initialization

For 'train' and 'val' phases:

call the fetch and check routine 'n' number of times based on num of heads. This is the initial place to create multiple inputs to be passed to the model

call check dimensionality routine 'n' times similar to above.

Now add specific routine for your data before slicing (via slice builder). For Kinetochores use case I've performed dilation and euclidean distance transform separately for segmentation and regression heads respectively.

Get individual slice builders for n heads and set the label slices accordingly.

In __getitem__ method, get individual ids for each slice and perform transformations per slice.

IMPORTANT: Original script returns raw and label slices, HERE: return raw and all the slices as per number of heads.

utils.py contains slicebuilder class and other properties.
In the initialization

call build slices as per the number of heads. Define the properties for slices of each head.


Model - model.py (multihead function, defining 'n' heads in init for model)

Here, add a function defining your multihead model.

In Abstract3DUnet initialization, define each head individually before defining if the task is segmentation or regression.

In forward method:
Get the 'n' heads for the output of decoder and before applying the final activation.
Return 'n' outputs from the forward routine.

Losses - losses.py (multihead loss: combining the segmenatation and regression loss)

In the `_AbstractDiceLoss`, modify the forward function based on the number of heads.
Define a multihead loss class with based `_AbstractDiceLoss` class.
Add the individual loss functions as per your case and define the forward accordingly.

Metric - Currently combining the MeanIoU and RMSE
Define a new metric e.g. 'Multihead metric' based on your case.

Trainer and Predictor - Get 2 outputs and not applying final activation (passing final activation as None in config)

In train routine, get 'n' targets from split_training_batch function and pass these to the forward pass

In predict routine, after setting the eval mode, get 'n' predictions out of the the forward pass. Just below this, concatenate all the
predictions as a list.

DONE. 