# An Unofficial Implementation to Combination of HuMoR and PARE 

## background 

[HuMoR](https://github.com/davrempe/humor) can recovery a smooth human mesh from videos. The official code has implemented 3d point clouds, 2d keypoints pipeline. However, currently most of human pose estimation method can only output root-relatived results instead of 3d point clouds in real world. 


## Method

We use the root aligned coordinate as the optimization target to solve this problem. We use PARE as the estimator.


This code is highly dependent on [PARE](https://github.com/mkocabas/PARE) and [HuMoR](https://github.com/davrempe/humor)