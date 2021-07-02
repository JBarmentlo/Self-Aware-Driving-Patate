from ..config import sim_config, net_config
from . import PortHandler, Sim
import logging

SimHandlerLogger = logging.getLogger("SimHandler ")
SimHandlerLogger.setLevel(logging.DEBUG)


class SimHandler():
    def __init__(self, max_concurrent_sims = sim_config.max_concurrent_sims, start_port = net_config.start_port):
        '''
            Will handle Sim creation, verification and kills.
            Args:
                max_concurrent_sims ([int], optional): [Maximum number of sims allowed to run at the same time]. Defaults to sim_config.max_concurrent_sims.
                start_port ([int], optional): [First port to use for the simulators]. Defaults to net_config.start_port.
        '''
        self.max_concurrent_sims = max_concurrent_sims
        self.porthandler = PortHandler(start_port = start_port, nb_ports=max_concurrent_sims)
        self.sims = {}
        SimHandlerLogger.debug(f"Started Simhandler with {self.max_concurrent_sims =}")

    
    def  can_i_launch_a_new_sim(self):
        if (len(self.sims) < self.max_concurrent_sims):
            return True
        return False


    def sanity_check(self, port):
        if (port is None):
            SimHandlerLogger.error(f"The SimHandler has declared it could start a new sim but the PortHandler has no ports left")
        if (port in self.sims.keys()):
            SimHandlerLogger.error(f"A new Sim instance is being created but a Sim instance with the same port exists in self.sims")


    def start_new_sim(self):
        '''
            Start a new Sim and update the porthandler to declare the used port.

            Returns:
                port (int): port number of the launched simulator if succesfull
                None if no sim was launched
        '''
        SimHandlerLogger.debug(f"Trying to start new sim.")
        if (self.can_i_launch_a_new_sim()):
            port = self.porthandler.get_available_port()
            self.sanity_check(port)
            self.sims[port] = Sim(port)
            self.porthandler.status[port] = False
            SimHandlerLogger.debug(f"Sim started with port : {port}")
            return port
        else:
            SimHandlerLogger.debug("No ports were available")
        return None


    def kill_sim(self, port):
        '''
            Kills the donkeysim process with given port and remove Sim instance from self.sims
            Updates the porthandler to declare the freed port.
            Args:
                port ([int]): port number of Sim to kill.
        '''
        SimHandlerLogger.debug("Killing sim")
        try:
            SimHandlerLogger.debug(f"Sim: {self.sims[port]} to be killed")
            self.sims[port].kill()
            self.sims.pop(port)
            self.porthandler.status[port] = True
            SimHandlerLogger.debug("Successfully killed sim process")
        except Exception as e:
            SimHandlerLogger.error(f"Tried to kill sim that does not exist. Error:\n{e}")


    def kill_idle_sims(self):
        '''
            Kills all sims that timed out
        '''
        SimHandlerLogger.debug(f"Checking for idle sims. {sim_config.kill_on_timeout =}")
        if (sim_config.kill_on_timeout):
            sims_to_kill = []
            for port, sim in self.sims.items():
                if (sim.is_timeout()):
                    sims_to_kill.append(port)
            for port in sims_to_kill:
                SimHandlerLogger.debug("idle sim to kill:", sims_to_kill)
                self.kill_sim(port)

    
    def clean_dead_sims(self):
        '''
            Checks wether the Sim's donkeysim process is still running and removes the dead ones.
        '''
        dead_sims = []
        for port, sim in self.sims.items():
            if (not sim.is_alive()):
                dead_sims.append(port)
        for port in dead_sims:
            SimHandlerLogger.error("Dead sim processes (not killed by us): ", dead_sims)
            self.sims.pop(port)


    def ping_sim(self, port):
        SimHandlerLogger.debug("Pinging sim {port}")
        try:
            self.sims[port].ping()
        except:
            SimHandlerLogger.warning(f"Tried to ping unexisting Sim at port {port}")
