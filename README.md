# Kinetochores

Initial attempt at self supervised training for the volume generation with autoencoder and 3D UNet.

## Initial experiments:
1. Autoencoder + L2 loss for volume reconstruction.
2. 3D UNet + L2 loss for volume reconstruction.
3. 3D UNet + RMSLE loss for volume reconstruction.

## Data
Data can be downloaded from the link (drive link) provided in the [data/dataset.txt](https://github.com/kreshuklab/Kinetochores/blob/master/data/dataset.txt)

Volume: \[1, 1000, 1000, 61\] - (C, X, Y, Z) the signals are concentrated around center of volume i.e. 400-600 in X, Y and 20-52 in Z. 18 volumes across time.

Annotations: Center of Mass coordinates for 79 signals in total over the 3d space (considered separately for now instead of 40 pairs, Pair3_2 is missing -> 79).

Format - PairX\_1, PairX\_2 (X ranging from 1 to 40)

## New experiments:
1. Binary mask creation based on COM coordinates: dataset['raw'] and datatset['label'] with (1, 48, 128, 128) volume, label shape respectively. The configs are set in train\_config\_EGFP.yml
based on config direction in [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet)


## Update with new supervised experiments:
1. Supervised experiments with vanilla 3d unet (own version) and L2 loss, not really useful predictions.
2. Same network, addition of gaussian in the loss function, slightly improved results.
2. Pytorch-3dunet experiments with DICE loss -> highly improved but blurry boundaries in the results.
3. DICE loss + gaussian smoothing -> GaussianDiceLoss : Good qualitative results, can move for the quantitative analysis.
4. Working to postprocess the predictions comparing with the target for localization of the emitters.
5. Current approach : 3d Maxima output from predictions (called peaks) -> binarize peaks -> EDT on Target -> label func on both peaks and target -> get emitter positions -> match peaks to emitters from target

## Update on quantitative results:
1. Spatial distance euclidean and standard euclidean (can provide weights as per the variance across x,y,z). Resolution - x,y: 130nm, z: 300nm

## Next steps:
1. Create volumes with the matched and unmatched label coordinates. *No data augmentation yet. Possible unmatched issues due to intensity difference, but need check.*

## Update 25/08:
1. Reduction of gaussian kernel across time/ epochs during training procedure implemented. Current approach involves epoch based kernel values. Need to automate this for random kernel assignment for possibly better results.
2. File with incorrect labels in Z across time added.
3. Next -> Passing multiple volumes with temporal context (3 at once for now).

## Update 01/09:
1. Temporal context done, 3 volume as 3 channels -> Need cluster for this, fails in memory locally.
2. Current experiments -> 
a) Temporal vols + Dice 
b) Temporal vols + GaussianDice (Fixed gaussian) -> b.1] k, sigma: (3,1) b.2] (5,1) b.3] (7,2)
c) Temporal vols + Reduction in kernel, sigma over time.

## Update 07/09:
1. There are some labels assigned as the medium intensity sources, which seem to be there but if compared to the high intensity sources, its a bit different. So in reconstruction, we need to have that blur area around the high intensity blob to be present.
2. Many labels seem to be misaligned and thus can be corrected but this can't be a major issue since its labeling in 3d and the peak finder can pick these points. *Thus thresholding value is important.*

## Experiments:
Data -> 1 channel raw, 1 channel target
1. Vanilla 3d UNet + RMSE loss
2. Vanilla 3d UNet + Dice loss
3. PyTorch 3d UNet + MSE loss
4. PyTorch 3d UNet + Dice loss
5. PyTorch 3d UNet + GaussianDice loss -> a) (3,1) b) (5,1) c) (7,2) d) (11,3)

6. Decreasing gaussian + dice loss -> Kernel, sigma values (7,2) -> (5,1) -> (3,1) *** 6 epochs only
Need to try with more epochs and gaussian values

Data -> 2 channel (EGFP + mcherry) raw, 1 channel target
7. PyTorch 3d UNet + Dice loss
8. PyTorch 3d UNet + GaussianDice loss -> a) (3,1) b) (5,1) c) (7,2) d) (11,3) e) (5,2) f) (14,4)

Data -> 3 channel (EGFP only with t-1,t,t+1) raw, 1 channel (t) target
9. PyTorch 3d UNet + Dice
10. PyTorch 3d UNet + GaussianDice -> 10.1] (3,1) a) 10 epochs b) 20 epochs, 10.2] (5,1) a) 10 epochs b) 20 epochs, 10.3] (7,2) a) 10 epochs b) 20 epochs

Data -> 6 channel (EGFP + mcherry: [t-1(eg), t-1(mc), t(eg), t(mc), t+1(eg), t+1(mc)]) raw, 2 channel (t) target
11. PyTorch 3d UNet + Dice loss
12. PyTorch 3d UNet + GaussianDice loss -> a) (3,1) b) (5,1) c) (7,2) d) (14,4) e) (9,3) f) (5,2)

We tried using dilated labels with Dice loss and gaussian dice loss but need to reuse the idea with the temporal setting as it can be useful here specifically.

## Update 17/10:
Regression approach: apply distance transform on the labels, network regression for distance to minimum in a region. Then apply peak local min to get all the minima and the neighborhood.
Experiments - 1 channel, 2 channel, 3 channel and 6 channel

1. Instance segmentation with stardist -> distance transform on the labels, apply thresholding to get spherical labels, relabel neighborhood based on original label coordinates. Stardist grid and rays parameters are the key.

### Stardist experiments:
We have the polyhedra fitting function. It works for 3d example data with 48 rays and anisotropy of (2,1,1) in Z,Y,X. 
Label issues - resolved. Normal range anisotropy. Now 1024 rays fits with 0.9 probability but its large number. Reduce the num of rays and check which works better.

Results with normal vols and temporal vols.

In normal vols, raw and labels: [48,128,128]
In temporal vols, raw data: [48,128,128,3] and labels: [48,128,128] - only the centre vol label.


## Update  24/10:
New metric for matching the prediction peaks with the dilated labels implemented in training. the metric returns the number of peaks matched.


## Update 28/10:
The actual useful IoU metric implemented. Taking threshold of predictions, getting local peaks -> dilate the peaks, remove the peaks below threshold.


## Update 30/10:
1. Train all the experiments with this new IoU metric. - the metric results are not really great, which is expected. Move to soft IoU or mAP.
2. Single channel working better than temporal thing, basically we are getting much dense predictions and higher num of output peaks.


1. Optimize the NMS thresholds in stardist - Done.
2. Eval metric done with segmentation output reaching 69 TP, 10 FN and 17 FP, which is good for temporal vol 5 (label vol 13).
3. Regression postprocessing involves either a) take the prediction output -> apply gaussian filter and take minima, get watershed and matching. b) Invert the output and apply gaussian filter and proceed with segmentation like postprocessing. - Done

## ToDo:
2. Get the StarDist output -> F1 score.

## Update 13/11:
1. Temporal dice loss added.
2. TODO: Add distribution learning loss


## WIP:
2. Need the soft IoU or mAP running.
3. Prepare the complete pipeline starting with the input vol and final segmentation output.

## Another thing to try (let's keep this aside for now):
Harmonic embeddings are there for the 2d data and it works really good with the instance segmentation of biological images. The idea could be to adapt it to 3d data.
Harmonic embeddings network are based on 2d datasets -> 1. quick trial with 2d slices for kinetochores data. (Not really useful)
In fact, let's try -> 2.[Single cell](https://github.com/opnumten/single_cell_segmentation) and 3.[Spatial embeddings](https://github.com/davyneven/SpatialEmbeddings) both on 2d slices for our data.


