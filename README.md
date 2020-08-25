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
