# Deep ACE
<p align="justify">
Cochlear implant (CI) users struggle to understand speech in noisyconditions.   In  this  work,  we  propose  an  end-to-end  speech  cod-ing  and  denoising  sound  coding  strategy  that  estimates  the  elec-trodograms  from  the  raw  audio  captured  by  the  microphone.   Wecompared this approach to a classic Wiener filter and TasNet to as-sess  its  potential  benefits  in  the  context  of  electric  hearing.   Theperformance of the network is assessed by means of noise reduc-tion performance (signal-to-noise-ratio improvement) and objectivespeech intelligibility measures.   Furthermore,  speech intelligibilitywas measured in 5 CI users to assess the potential benefits of eachof the investigated algorithms.  Results suggest that the speech per-formance of the tested group seemed to be equally good using ourmethod compared to the front-end speech enhancement algorithm.

<p align="center">
  <img src="fig.png"  alt="70%" width="70%"/>
</p>

# Requirements
See [Requirements.txt](https://github.com/APGDHZ/DeepACE/requirements.txt)

# References
<p align="justify">
[1] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.

[2] Luo Y, Mesgarani N. TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation. arXiv preprint arXiv:1809.07454, 2018.
