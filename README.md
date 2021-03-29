# Self-Aware-Driving-Patate

A project by @dberger @jbarment @ldevelle @llenotre 

# Usage:

From base directory (or model save and load is broken):
```sh
python3 Archi/train_simulator.py --sim ../../DonkeyCar/DonkeySimLinux/donkey_sim.x86_64 --model 'new_model.h5'
```

# Architecture

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
  
# Utils

- Multiprocessing -> Takes complete architecture in hand
- Qarnot computing -> HyperParams optim : https://github.com/ezalos/Qarnot_Wrapper
- Distance control of WS : https://github.com/ezalos/emails
  
