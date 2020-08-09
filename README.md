# SB_RL_Trading
Training agents with StableBaselines to trade on Stock Market

This is repository for article about Deep Reinforcement Learning. Example code shows how to build an Agent that can learn how to trade on a Stock Market without any prior knowledge. It basically figures it out itself only receiving a reward signal that indicates if chosen action was good or bad.

###Installation requirements:

- tensorflow==1.15.3
- tesorboard==1.15.0
- stable-baselines==2.10.1
- gym==0.17.2
- pandas==1.1.0
- scikit-learn==0.23.2

note: Tensorflow needs to be in 1.x version in order to be compatible with Stable Baselies 

###Tensorboard configuration

All the training and evaluation results are saved and can be show using tensorboard. To start a tensorboard server use this command:

```bash
 tesorboard --logdir ./tensorboard
```