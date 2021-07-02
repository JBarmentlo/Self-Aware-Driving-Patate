from ..config import sim_config, net_config
import logging
# TODO: check for port conflicts with PrivateAPI port

PortLogger = logging.getLogger("PortHandler ")
PortLogger.setLevel(logging.DEBUG)


class PortHandler():
    def __init__(self, start_port = net_config.start_port, nb_ports = sim_config.max_concurrent_sims):
        '''
            An object for handling ports.
            self.status: True means available, False means in use.

            Args:
                start_port (int, optional): [description]. Defaults to 9090.
                nb_ports ([type], optional): [description]. Defaults to simconfig.max_concurrent_sims.
        '''
        self.status = self.create_status(start_port, nb_ports)
        PortLogger.debug(f"Created PortHandler with status: {self.status} and start port: {start_port}")


    def create_status(self, start_port, nb_ports):
        status = {}
        for i in range(nb_ports):
            status[start_port + i] = True
        return status

    
    def is_port_availabe(self, port_no):
        '''
            Returns True is port_no is available and within the allowed range

            Args:
                port_no ([int]): requested port number

            Returns:
                [bool]: Availability
        '''
        PortLogger.debug(f"Checking availability for port: {port_no}")
        try:
            return (self.status[port_no])
        except Exception as e:
            PortLogger.error("Error occured: {e}")
            return False

    
    def get_available_port(self) -> int or None:
        '''
            Returns:
                int: Available port, None if no port is available
        '''
        for key, value in self.status.items():
            if (value):
                return(key)
        return (None)


    def number_of_available_ports(self):
        i = 0
        for value in self.status.values():
            if (value):
                i += 1
        return i