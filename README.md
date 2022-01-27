# Deep ACE
<p align="justify">
Cochlear implant (CI) users struggle to understand speech in noisy conditions. To address this problem, we propose a deep learning speech denoising sound coding strategy that estimates the CI electric stimulation patterns out of the raw audio data captured by the microphone, performing end-to-end CI processing. To estimate the relative denoising performance differences between various approaches, we compared this technique to a classic Wiener filter and to a conv-TasNet. Speech enhancement performance was assessed by means of signal-to-noise-ratio improvement and the short-time objective speech intelligibility measure. Additionally, 5 CI users were evaluated for speech intelligibility in noise to assess the potential benefits of each algorithm. Our results show that the proposed method is capable of replacing a CI sound coding strategy while preserving its general use for every listener and performing speech enhancement in noisy environments, without sacrificing algorithmic latency.

<p align="center">
  <img src="fig.png"  alt="60%" width="60%"/>
</p>

# Requirements
See [Requirements.txt](requirements.txt)

# References
<p align="justify">
[1] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.

[2] Y. Luo and N. Mesgarani, “Conv-TasNet: Surpassing ideal time–frequency magnitude masking for speech separation,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 27, pp. 1256–1266, 2019.
