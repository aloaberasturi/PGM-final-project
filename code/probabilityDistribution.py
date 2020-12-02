class ProbabilityDistribution():
    """
    Class for probability distribution objects
    """
    def __init__(self, support, values):
        self.support = support
        self.values = values
    
    def check_integrity(self):
        # check if all values sum up to one 
