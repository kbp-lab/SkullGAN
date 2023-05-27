# SkullGAN

SkullGAN is a novel generative adversarial network (GAN) for the generation of synthetic computed tomography (CT) slices of the human skull. It has been developed to address the prevalent bottleneck in machine learning healthcare applications - the scarcity of carefully curated medical images.

Paper: [arXiv](https://arxiv.org/list/cs.AI/recent)
Project Page: [KBP Lab](https://kbplab.stanford.edu/SkullGAN/)

## Abstract

Machine learning holds substantial potential for diverse healthcare applications. Some applications involving the human skull include segmentation, reconstruction, anomaly detection, denoising, and transcranial focused ultrasound simulations. However, the broad adoption of deep learning for these uses necessitates large volumes of carefully curated medical images to train effective and generalizable models. 

Our solution, SkullGAN, addresses this problem by generating a vast dataset of synthetic CT slices of the human skull. The generated synthetic skulls were first validated based on three quantitative radiological features: skull density ratio (SDR), mean thickness, and mean intensity. However, we show that these metrics can be easily fooled and therefore evaluate SkullGAN using t-distributed stochastic neighbor embedding (t-SNE) and by applying the trained discriminator as a classifier. 

The results show that SkullGAN is capable of generating large numbers of synthetic skull CT segments that are indistinguishable from real skull CT segments, and therefore alleviates obstacles such as access, capital, time, and domain expertise in preparing high quality training datasets. SkullGAN makes it possible for any researcher to generate thousands of highly varied skull CT segments in a few minutes and at a very low cost, for training neural networks with medical applications that involve the human skull.

## Getting Started

(Fill with installation instructions and usage guide.)

## Examples

(Fill with usage examples and visualisations.)

## Citation



## Contact

(Fill with contact info.)
