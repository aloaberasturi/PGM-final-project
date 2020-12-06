#!/usr/bin/python3
import json
class ProbabilityDistribution():
    """
    Class for probability distribution objects
    """
    def __init__(self, support, probability=None):
        self.support = support
        if (probability != None):
            self.probability = {sample: p for sample, p in zip(self.support, probability)}
        # else:
        #     self.probability = {sample : 1.0/len(self.support)  if self.is_uniform else 0.0 for sample in self.support}
        self.calculate_min_max_support_values()


    def calculate_min_max_support_values(self):
        try:
            self.max_support_value = max(self.support)
            self.min_support_value = min(self.support)
        except ValueError:
            print("The support of the distribution is empty")
    
    def add_sample(self, sample, prob):
        # We have a discrete probability distribution, 
        # so we map the sample to the corresponding slot
        sample = self.map_sample(sample)
        self.probability[sample] = prob
    
    def map_sample(self, sample):
        if (sample < self.min_support_value * 2 or sample > self.max_support_value * 2):
            raise ValueError(
                            """ \n Error mapping a sample in the discrete probability distribution.
                            Samples cannot be out of the range: %d and %d" % 
                            (self.min_support_value * 2, self.max_support_value * 2) """
                        )
        if (sample < self.min_support_value):
            return self.min_support_value
        
        if (sample > self.max_support_value):
            return self.max_support_value
        
        return round(sample)

    def get_support(self):
        return self.support

    def get_prob(self, sample):
        sample = self.map_sample(sample)

        if (sample in self.probability.keys()):
            return self.probability[sample]
        return -1

    def get_probs(self):        
        probs = [self.probability[sample] for sample in self.get_support()]
        return probs

    def check_integrity(self):
        sum = 0.0
        for prob in self.probability.values():
            if (prob < 0.0 or prob > 1.0):
                return False
            sum += prob
        
        return abs(1.0 - sum) < 0.00000001

    def __str__(self):
        """
        Overrides __str__() method
        """
        string = "Probability distribution: " + json.dumps(self.probability)
        return string
   
