# SimLauncher3000
Launch Donkey Car Simulation.


Use as a module (intended use):    
from simlauncher3000 import Client, Server, start_server    
then use functions as described below


launch the server :    
```bash
source .env
python test_server.py
```

Use the client   
```python
from modules import Client

c = Client()
c.request_simulator()
c.ping_sim()
c.kill_sim()
```
