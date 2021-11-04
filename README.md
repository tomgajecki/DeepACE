# Deep ACE
<p align="justify">
Cochlear implant (CI) users struggle to understand speech in noisy conditions.   In  this  work,  we  propose  an  end-to-end  speech  cod-ing  and  denoising  sound  coding  strategy  that  estimates  the  electrodograms  from  the  raw  audio  captured  by  the  microphone.   We compared this approach to a classic Wiener filter and TasNet to assess  its  potential  benefits  in  the  context  of  electric  hearing.   The performance of the network is assessed by means of noise reduction performance (signal-to-noise-ratio improvement) and objective speech intelligibility measures.   Furthermore,  speech intelligibility was measured in 5 CI users to assess the potential benefits of each of the investigated algorithms.  Results suggest that the speech performance of the tested group seemed to be equally good using our method compared to the front-end speech enhancement algorithm.

<p align="center">
  <img src="fig.png"  alt="70%" width="70%"/>
</p>

# Requirements
See [Requirements.txt](requirements.txt)

# References
<p align="justify">
[1] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.

[2] Luo Y, Mesgarani N. TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation. arXiv preprint arXiv:1809.07454, 2018.
