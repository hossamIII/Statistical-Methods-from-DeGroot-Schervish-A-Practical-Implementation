import numpy as np
# input data 
Weights = [(65, 84),(85,104),(105, 124),(125, 144),(145, 164),(165, 184),(185, 204)]
Frequencies = [9, 10, 17, 10, 5, 4, 5]

class CentralTendency:
    def __init__(self, Weights, Frequencies):
        self.Wg = np.array(Weights)   # to be able to use the data for other functions
        self.Fq = np.array(Frequencies)
        self.midpoint = (self.Wg[:,0] + self.Wg[:,1])/2
        self.continuous = np.argmax(self.Fq) > 0 and self.Wg[np.argmax(self.Fq) - 1][1] == self.Wg[np.argmax(self.Fq)][0]
        self.w = self.Wg[np.argmax(self.Fq)][1] - self.Wg[np.argmax(self.Fq)][0] + (1 if not self.continuous else 0)

    def ArithmeticMean(self):
        
        return sum(self.midpoint * self.Fq) / sum(self.Fq)
    def GeometricMean(self):
        return np.exp(sum(np.log(self.midpoint) * self.Fq) / sum(self.Fq))
    def HarmonicMean(self):
        return sum(self.Fq)/sum(self.Fq/self.midpoint)
    def Mode(self):
        modal_index =np.argmax(self.Fq)
        # Lower boundary of the modal class
        Lmo = self.Wg[modal_index][0]
        # Class width 
        w = self.Wg[modal_index][1] - self.Wg[modal_index][0] + (1 if not self.continuous else 0)

        Delta1 = self.Fq[modal_index] if modal_index == 0 else self.Fq[modal_index] - self.Fq[modal_index - 1]
        Delta2 = self.Fq[modal_index] if modal_index == len(self.Fq) - 1 else self.Fq[modal_index] - self.Fq[modal_index + 1]        
        return Lmo + self.w * (Delta1 / (Delta1 + Delta2))
    def Median(self):
        median_pos = sum(self.Fq) / 2
        median_index = np.argmax(np.cumsum(self.Fq) >= median_pos) # Finds the index where the cumulative frequency first reaches or exceeds 30, identifying the median class.
        Lmed = self.Wg[median_index][0]
        Fmed = self.Fq[median_index]
        Cf = sum(self.Fq[:median_index]) if median_index > 0 else 0  # Cumulative freq up to previous class
        return Lmed + (self.w /Fmed) * ((sum(self.Fq)/2) - Cf) 
    def Quartiles(self):
        i = int(input("Enter which quartile (1, 2, or 3):"))
        if not (1 <= i <= 3):
            raise ValueError("Enter a number between 1 and 3")
        quartile_pos = (i/ 4) * sum(self.Fq)   
        quartile_index = np.argmax(np.cumsum(self.Fq) >= quartile_pos)    
        Lq = self.Wg[quartile_index][0]
        Fq = self.Fq[quartile_index]
        Cf = sum(self.Fq[:quartile_index]) if quartile_index > 0 else 0  # Cumulative freq up to previous class
        return " Your " + str(i) +"st quartile is " + str(Lq + (self.w/ Fq) *((i/ 4) * sum(self.Fq) - Cf))
    def Deciles(self):
        i = int(input("Enter which Deciles (1, 2 .. 9):"))
        if not (1 <= i <= 9):
            raise ValueError("Enter a number between 1 and 9")
        Percentiles_pos = (i/ 10) * sum(self.Fq)   
        Decile_index = np.argmax(np.cumsum(self.Fq) >= Percentiles_pos)    
        Lq = self.Wg[Decile_index][0]
        Fq = self.Fq[Decile_index]
        Cf = sum(self.Fq[:Decile_index]) if Decile_index > 0 else 0  # Cumulative freq up to previous class
        return " Your " + str(i) +"st decile is " + str(Lq + (self.w/ Fq) *((i/ 10) * sum(self.Fq) - Cf))
    def Percentiles(self):
        i = int(input("Enter which Deciles (1, 2 .. 99):"))
        if not (1 <= i <= 99):
            raise ValueError("Enter a number between 1 and 99")
        Percentiles_pos = (i/ 100) * sum(self.Fq)   
        Percentiles_index = np.argmax(np.cumsum(self.Fq) >= Percentiles_pos)    
        Lq = self.Wg[Percentiles_index][0]
        Fq = self.Fq[Percentiles_index]
        Cf = sum(self.Fq[:Percentiles_index]) if Percentiles_index > 0 else 0  # Cumulative freq up to previous class
        return " Your " + str(i) +"st percentile is " + str(Lq + (self.w/ Fq) *((i/ 100) * sum(self.Fq) - Cf))
Ct = CentralTendency(Weights, Frequencies)
print(Ct.ArithmeticMean())
print(Ct.GeometricMean())   
print(Ct.HarmonicMean()) 
print(Ct.Mode())
print(Ct.Median())
print(Ct.Quartiles())
print(Ct.Deciles())
print(Ct.Percentiles())