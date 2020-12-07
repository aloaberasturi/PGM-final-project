#!/usr/bin/python3
import json
class ProbabilityDistribution():
    """
    Class for probability distribution objects
    """
    def __init__(self, variable, consequent=None, probabilities=None):
        self.support = variable.support
        self.consequent = consequent
        self.calculate_min_max_support_values()
        if (probabilities != None):
            self.probabilities = {sample: p for (sample, p) in zip(self.support, probabilities)}
        self.check_integrity()
    def calculate_min_max_support_values(self):
        try:
            self.max_support_value = max(self.support)
            self.min_support_value = min(self.support)
        except ValueError:
            print("The support of the distribution is empty")
    
    def add_sample(self, sample, prob):
        self.probabilities[sample] = prob    

    def get_support(self):
        return self.support

    def get_prob(self, sample):
        if (sample in self.probabilities.keys()):
            return self.probabilities[sample]
        raise ValueError("The given sample does not match any value in the distribution's support")

    def get_probs(self):        
        return [self.probabilities[sample] for sample in self.get_support()]

    def check_integrity(self):
        sum = 0.0
        for prob in self.probabilities.values():
            if (prob < 0.0 or prob > 1.0):
                raise ValueError("Probability distribution is ill-formed")
            sum += prob
        
        return abs(1.0 - sum) < 0.00000001

    def __str__(self):
        """
        Overrides __str__() method
        """
        string = "Probability distribution: " + json.dumps(self.probabilities)
        return string
   
