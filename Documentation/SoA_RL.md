# Reinforcement Learning

## General principle

![Genral](https://spinningup.openai.com/en/latest/_images/rl_diagram_transparent_bg.png)

## The RL problem

We want to optimize the equation `J(pi)` so we get the maximum return:

![](https://spinningup.openai.com/en/latest/_images/math/f0d6e3879540e318df14d2c8b68af828b1b350da.svg)


## The Bellman Equations

They allow us to update our estimation of the expected total reward 

![](https://spinningup.openai.com/en/latest/_images/math/7e4a2964e190104a669406ca5e1e320a5da8bae0.svg)

## The different types of RL Algorithms

https://spinningup.openai.com/en/latest/spinningup/rl_intro.html

![](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)

We will focus ourselves on the Soft Actor Critic and Deep Q-Network

### SAC

https://spinningup.openai.com/en/latest/algorithms/sac.html

Algorithm:

![](https://spinningup.openai.com/en/latest/_images/math/c01f4994ae4aacf299a6b3ceceedfe0a14d4b874.svg)

### DDQN

https://paperswithcode.com/method/double-dqn

Original paper: https://arxiv.org/pdf/1509.06461v3.pdf

The double Q-learning update step:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/4941acabf5144d1b3e9c271606011abdc0df444d)
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/3e37476013126ddd4afdba69ef7b03767f4c4b75)

$$
 Y^{DoubleDQN}_{t} = R_{t+1}+\gamma{Q}\left(S_{t+1}, \arg\max_{a}Q\left(S_{t+1}, a; \theta_{t}\right);\theta_{t}^{-}\right) 
$$
