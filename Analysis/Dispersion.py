import numpy as np
from central_tendency import CentralTendency  

# Input data: weight intervals and their corresponding frequencies
Weights = [(65, 84), (85, 104), (105, 124), (125, 144), (145, 164), (165, 184), (185, 204)]
Frequencies = [9, 10, 17, 10, 5, 4, 5]

class Dispersion:
    def __init__(self, Weights, Frequencies):
        # Initialize with CentralTendency instance to reuse midpoint and statistical methods
        self.ct = CentralTendency(Weights, Frequencies)
        
        # Save the data attributes from CentralTendency for reuse
        self.Wg = self.ct.Wg                # Class intervals as NumPy array
        self.Fq = self.ct.Fq                # Frequencies as NumPy array
        self.midpoints = self.ct.midpoint   # Midpoints of each class
        self.total_freq = sum(self.Fq)      # Total number of data points

    def Range(self):
        # Range = max value - min value in dataset
        return np.max(self.Wg) - np.min(self.Wg)

    def RelativeRange(self):
        # Relative Range = (max - min) / (max + min), gives a normalized spread
        return (np.max(self.Wg) - np.min(self.Wg)) / (np.max(self.Wg) + np.min(self.Wg))

    def QuartileDeviation(self):
        # Quartile Deviation = (Q3 - Q1) / 2, a robust measure of dispersion
        return (self.ct.Quartiles(3) - self.ct.Quartiles(1)) / 2

    def MDMean(self):
        # Mean Deviation about Mean: Σ|xi - mean| * fi / Σfi
        deviations = np.abs(self.midpoints - self.ct.ArithmeticMean()) * self.Fq
        return sum(deviations) / sum(self.Fq)

    def MDMedian(self):
        # Mean Deviation about Median: Σ|xi - median| * fi / Σfi
        deviations = np.abs(self.midpoints - self.ct.Median()) * self.Fq
        return sum(deviations) / sum(self.Fq)

    def MDMode(self):    
        # Mean Deviation about Mode: Σ|xi - mode| * fi / Σfi
        deviations = np.abs(self.midpoints - self.ct.Mode()) * self.Fq
        return sum(deviations) / sum(self.Fq)

    def CoefficientMD(self):
        # Coefficient of Mean Deviation = MD / respective central value
        # Useful to compare dispersion relative to mean, median, mode
        CMDMean = self.MDMean() / self.ct.ArithmeticMean()
        CMDMedian = self.MDMedian() / self.ct.Median()
        CMDMode = self.MDMode() / self.ct.Mode()
        return CMDMean, CMDMedian, CMDMode

    def PopulationVariance(self):
        # Population Variance: Σfi(xi - μ)² / Σfi
        deviations = self.midpoints - self.ct.ArithmeticMean()  
        squared_deviations = np.power(deviations, 2) * self.Fq
        return np.sum(squared_deviations) / np.sum(self.Fq)

    def SampleVariance(self):
        # Sample Variance: Σfi(xi - x̄)² / (Σfi - 1)
        deviations = self.midpoints - self.ct.ArithmeticMean()  
        squared_deviations = np.power(deviations, 2) * self.Fq
        return np.sum(squared_deviations) / (np.sum(self.Fq) - 1)

    def StandardDeviation(self):
        # Returns both Population and Sample Standard Deviation (square root of variance)
        return np.sqrt(self.PopulationVariance()), np.sqrt(self.SampleVariance())

# Create instance of Dispersion class
Ds = Dispersion(Weights, Frequencies)

# Display different dispersion measures with clear labels
print("Range:", Ds.Range())
print("Relative Range:", Ds.RelativeRange())
print("Quartile Deviation (Semi-IQR):", Ds.QuartileDeviation())

print("Mean Deviation about Mean:", Ds.MDMean())
print("Mean Deviation about Median:", Ds.MDMedian())
print("Mean Deviation about Mode:", Ds.MDMode())

# Coefficient of Mean Deviation (returns a tuple)
cmd_mean, cmd_median, cmd_mode = Ds.CoefficientMD()
print("Coefficient of Mean Deviation (Mean):", cmd_mean)
print("Coefficient of Mean Deviation (Median):", cmd_median)
print("Coefficient of Mean Deviation (Mode):", cmd_mode)

print("Population Variance:", Ds.PopulationVariance())
print("Sample Variance:", Ds.SampleVariance())

# Standard Deviation (returns a tuple)
std_pop, std_sample = Ds.StandardDeviation()
print("Population Standard Deviation:", std_pop)
print("Sample Standard Deviation:", std_sample)

