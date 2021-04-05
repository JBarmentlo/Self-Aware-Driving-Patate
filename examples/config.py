class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

cte_config = DotDict()
cte_config.cte_offset = 2.25
cte_config.max_cte = 3.2
cte_config.done_func = lambda x: abs(x) > cte_config.max_cte
