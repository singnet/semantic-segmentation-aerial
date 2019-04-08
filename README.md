[issue-template]: ../../issues/new?template=BUG_REPORT.md
[feature-template]: ../../issues/new?template=FEATURE_REQUEST.md

<a href="https://singularitynet.io/">
<img align="right" src="./docs/assets/logo/singularityNETblue.png" alt="drawing" width="160"/>
</a>

# Super Resolution

> Repository for SingularityNET's "Semantic Segmentation for Aerial Images" service

[![Github Issues](https://img.shields.io/github/issues-raw/singnet/semantic-segmentation-aerial.svg?style=popover)](https://github.com/singnet/semantic-segmentation-aerial/issues)
[![Pending Pull-Requests](https://img.shields.io/github/issues-pr-raw/singnet/semantic-segmentation-aerial.svg?style=popover)](https://github.com/singnet/semantic-segmentation-aerial/pulls)
[![GitHub License](	https://img.shields.io/github/license/singnet/dnn-model-services.svg?style=popover)](https://github.com/singnet/semantic-segmentation-aerial/blob/master/LICENSE)
[![CircleCI](https://circleci.com/gh/singnet/semantic-segmentation-aerial.svg?style=svg)](https://circleci.com/gh/singnet/semantic-segmentation-aerial)

This service uses convolutional neural networks to segment aerial images into:

```angular2
- Impervious surfaces (white)
- Buildings (blue)
- Low vegetation (cyan)
- Trees (green)
- Cars (yellow)
- Clutter (red)
- Undefined (black)
```

This repository was forked from [nshaud/DeepNetsForEO](https://github.com/nshaud/DeepNetsForEO). The original code is written in Python 3 (using Pytorch).

Refer to:
- [The User's Guide](https://singnet.github.io/semantic-segmentation-aerial/): for information about how to use this code as a SingularityNET service;
- [The Original Repository](https://github.com/nshaud/DeepNetsForEO): for up-to-date information on [nshaud](https://github.com/nshaud) implementation of this code.
- [SingularityNET Wiki](https://github.com/singnet/wiki): for information and tutorials on how to use the SingularityNET and its services.

## Contributing and Reporting Issues

Please read our [guidelines](https://github.com/singnet/wiki/blob/master/guidelines/CONTRIBUTING.md#submitting-an-issue) before submitting an issue. If your issue is a bug, please use the bug template pre-populated [here][issue-template]. For feature requests and queries you can use [this template][feature-template].

## Authors

* **Ramon Durães** - *Maintainer* - [SingularityNET](https://www.singularitynet.io)

## Licenses

This project is licensed under the MIT License. The original repository is licensed under the GNU General Public License. See the [LICENSE](LICENSE) file for details. 