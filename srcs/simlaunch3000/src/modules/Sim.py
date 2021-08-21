import logging
import subprocess
import os
import time
import shlex
from ..config import sim_config




class Sim():
    def __init__(self, port, time_till_timeout = sim_config.time_till_timeout):
        '''
            A Sim object starts and holds the donkeysim subprocess.
            It must be pinged every time_till_timeout seconds with self.keep_alive_ping() or self.is_timeout() will return True

            WARNING: IF THE PORT IS UNAVAILABLE NO ERROR IS THROWN AND A DONKEYSIM SUBPROCESS WILL START WITH UNKNOWN PORT

            Args:
                port ([int]): port number the sim will listen to. Must be available (use the PortHandler class)
                time_till_timeout (int, optional): Time until timeout without pings. Defaults to sim_config.time_till_timeout.
        '''
        self.port = int(port)
        self.SimLogger = logging.getLogger(f"SimProcess {self.port}")
        self.SimLogger.setLevel(logging.DEBUG)
        self.start_time = time.time()
        self.last_ping_time = time.time()
        self.time_till_timeout = time_till_timeout
        self.process = self.start_sim_proc()
        self.is_in_use = True




    def start_sim_proc(self):
        '''
            Starts the donkeysim subprocess using the self.port argument and returns the subprocess instance
        '''
        self.SimLogger.debug(f"Launching sim at port {self.port}")
        cmd = os.environ["SIM_PATH"] + " --port " + str(self.port)
        print(shlex.split(cmd))
        proc = subprocess.Popen(shlex.split(cmd))
        self.SimLogger.debug(f"Launched sim at port {self.port}")
        return proc


    def ping(self):
        '''
            The Sim instance must be pinged at least once every self.time_till_timeout seconds to stay alive.
        '''
        self.SimLogger.debug(f"Pinging sim at port {self.port}. Last ping time = {self.last_ping_time}")
        self.last_ping_time = time.time()

    
    def is_timeout(self):
        '''
            Checks if the process has been pinged to stay alive less that self.time_till_timeout seconds ago
        '''
        self.SimLogger.debug(f"Checking timeout for sim {self.port}")
        if (time.time() - self.last_ping_time) > self.time_till_timeout:
            self.SimLogger.debug(f"Sim found to be timed out with current time : {time.time()} and last_ping_time = {self.last_ping_time}")
            return True
        self.SimLogger.debug(f"not timed out")
        return False


    def is_alive(self):
        '''
            Returns True if the donkeysim process is still running.
        '''
        return (self.process.poll() == None)


    def kill(self):
        '''
            Kill the donkeysim subprocess and set self.process to None.
        '''
        self.SimLogger.debug(f"Killing {self}")
        self.process.kill()
        self.process = None


    def safety_check_port_arg():
        pass
        # TODO: Inquire as to the safety of this

    
    def __str__(self):
        return (f"Sim instance at port {self.port}")

    
    def __repr__(self):
        return str(self)


# if __name__ == "__main__":
#     sim = Sim(8889, 5)
#     # sim.permapoll()
#     while True:
#         time.sleep(1)
#         print(sim.is_alive())
#         if (not sim.should_be_alive()):
#             sim.kill_sim()

