# SkullGAN

SkullGAN is a novel generative adversarial network (GAN) for the generation of synthetic computed tomography (CT) slices of the human skull. It has been developed to address the prevalent bottleneck in machine learning healthcare applications - the scarcity of carefully curated medical images.

Paper: [arXiv](https://arxiv.org/list/cs.AI/recent) <br>
Project Page: [KBP Lab](https://kbplab.stanford.edu/SkullGAN/)

## Abstract

Machine learning holds substantial potential for diverse healthcare applications. Some applications involving the human skull include segmentation, reconstruction, anomaly detection, denoising, and transcranial focused ultrasound simulations. However, the broad adoption of deep learning for these uses necessitates large volumes of carefully curated medical images to train effective and generalizable models. 

Our solution, SkullGAN, addresses this problem by generating a vast dataset of synthetic CT slices of the human skull. The generated synthetic skulls were first validated based on three quantitative radiological features: skull density ratio (SDR), mean thickness, and mean intensity. However, we show that these metrics can be easily fooled and therefore evaluate SkullGAN using t-distributed stochastic neighbor embedding (t-SNE) and by applying the trained discriminator as a classifier. 

The results show that SkullGAN is capable of generating large numbers of synthetic skull CT segments that are indistinguishable from real skull CT segments, and therefore alleviates obstacles such as access, capital, time, and domain expertise in preparing high quality training datasets. SkullGAN makes it possible for any researcher to generate thousands of highly varied skull CT segments in a few minutes and at a very low cost, for training neural networks with medical applications that involve the human skull.

## Getting Started

We've included two ways to train SkullGAN, one through the command-line (source code located in 'cli/'), and one through a Jupyter notebook ('SkullGAN - Train.ipynb'). The Jupyter notebook is self-contained, and requires only that you have the necessary libraries and packages installed. While restrictions prevent us from uploading our training dataset, which consists of skull CT scans of real patients, we have included both our pre-trained model (Celeb-A), and the final SkullGAN model presented in the paper. These can be downloaded at this link: [SkullGAN PyTorch Models](https://drive.google.com/drive/folders/1KRLXFMssKKuQwXL5J9fVorhVGbZaSeK4?usp=sharing)



## Examples

(Fill with usage examples and visualisations.)

## Citation



## Contact

(Fill with contact info.)
