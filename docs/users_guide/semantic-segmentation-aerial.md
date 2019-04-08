[issue-template]: ../../../issues/new?template=BUG_REPORT.md
[feature-template]: ../../../issues/new?template=FEATURE_REQUEST.md

<!--
<a href="https://singularitynet.io/">
<img align="right" src="../assets/logo/singularityNETblue.png" alt="drawing" width="160"/>
</a>
-->

# Semantic Segmentation for Aerial Images

This service uses convolutional neural networks to segment aerial images, assigning each pixel to a label.

It is part of SingularityNET's third party services, [originally implemented by nshaud](https://github.com/nshaud/DeepNetsForEO).

### Welcome

The service takes as input:
- Required parameters:
    - input (string): the URL of an *infrared TIFF image* of interest;
    - window_size (integer): the size of the square sliding window used to perform the segmentation. The larger it is, the larger the image patch that will be processed at a time;
    - stride (integer): the amount by which to shift the sliding window. Its maximum is the size of the sliding window so as to cover the entire scope of the input image. The smaller it is, the higher the precision but also its processing time also increases significantly - usually `stride = window_size/2` generates good results. ;

### What’s the point?

This service can be used automatically segment aerial images into semantic labels following the color palette below:
```
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)
```

Hence this service can be used to estimate the amount or area of each of these items in the photo.

### How does it work?

You can use this service at [SingularityNET DApp](http://beta.singularitynet.io/) by clicking on `snet/semantic-segmentation-aerial`.

You can also call the service from SingularityNET CLI:

```
$ snet client call snet semantic-segmentation-aerial segment_aerial_image '{"input": "INFRARED_TIFF_IMAGE_URL", "window_size": "SLIDING_WINDOW_SIZE", "stride": "STRIDE"}'
```

Go to [this tutorial](https://dev.singularitynet.io/tutorials/publish/) to learn more about publishing, using and deleting a service.

### What to expect from this service?

Example:

**Input**

- input: "https://github.com/singnet/semantic-segmentation-aerial/raw/master/docs/assets/examples/top_mosaic_09cm_area27.tif"
- window_size: 512
- stride: 256

Input Image                          | Segmented Image
:-----------------------------------:|:-------------------------:
<img width="100%" src="assets/users_guide/top_mosaic_09cm_area27.png"> | <img width="100%" src="assets/users_guide/segmented_image.png">
