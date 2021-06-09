
# TODO: confirm memory object format


class Memory():
    def __init__(self):
        '''
            self.data should be an array of memory objects.
            in memory object format (state, action, reward, new_state, done, info) tuple
        '''
        self.data = ["data"]
        raise NotImplementedError

    
    def save_memory(self):
        '''
            Save the memory. (To s3, or local, or not at all)
        '''
        raise NotImplementedError
    

    def add_to_memory(self, memory_object):
        '''
            Adds memory object to self.data

            Args:
                memory_object (state, action, reward, new_state, done, info)
        '''
        # * idea : could add the object to self.newdata self.data so save_memory only adds the new memory to s3
        raise NotImplementedError


    def __str__(self):
        return f"Memory of size {len(self.data)}"


    def __repr__(self):
        return str(self)
