# Self-Aware-Driving-Patate

A project by @dberger @jbarment @ldevelle @llenotre @gilles595

A big thank to [Qarnot](https://qarnot.com/) who supports us through this endeaviour by offering us cloud computing.
If you would like to see how we interact with their platforn to launch our calculations, [Here's our wrapper repository](https://github.com/ezalos/Qarnot_Wrapper)

# Usage:

From base directory (or model save and load is broken):
```sh
python3 Archi/train_simulator.py --sim ../../DonkeyCar/DonkeySimLinux/donkey_sim.x86_64 --model 'new_model.h5'
```

# Architecture

![Architecture Overview](./Documentation/Overview.png)

1. Input data
   - Interaction with simulator
   - From datasets
  
2. Preprocessing
	- From raw data
  
3. Model training
	- reward optimisation
	- all hyper_parameters given in `init()`
	- followup of metrics (loss / accuracy)
  
4. Model evaluation
	- Saving the model, with HyperParams
	- Evaluate the model

$$ f(x, \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}}e ^\frac{-(x -\mu) ^ 2}{2\sigma ^ 2} $$

# Utils

- Multiprocessing -> Takes complete architecture in hand
- Qarnot computing -> HyperParams optim : https://github.com/ezalos/Qarnot_Wrapper
- Distance control of WS : https://github.com/ezalos/emails
  

# SimLauncher3000
Source the .env file, import Client, Server and start_server from the package.   
use
```python
c = Client()
c.request_simulator()
c.kill_sim()
```

# How to use with SimLauncher3000

## In docker

Where we train the agent:

```sh
export PS="wesh" ; python3.8 srcs --sim simlaunch3000 --model 'new_model.h5' --agent DDQN
```

## In computer

Where we run the simulator:

```sh
cd srcs/simlaunch3000
export PS="wesh" ; export SIM_PATH="/home/ezalos/Downloads/DonkeySimLinux/donkey_sim.x86_64" ; python3.8 test_server.py
```