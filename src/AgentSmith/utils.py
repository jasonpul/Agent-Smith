def log_decay_function(m: float, sd: float):
    return lambda x: 1 / (1 + (x / m) ** sd)
