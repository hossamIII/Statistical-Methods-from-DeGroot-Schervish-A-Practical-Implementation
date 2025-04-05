import numpy as np

# Class data: Weights are intervals (e.g., 65-84), Frequencies are counts in each interval
Weights = [(65, 84), (85, 104), (105, 124), (125, 144), (145, 164), (165, 184), (185, 204)]
Frequencies = [9, 10, 17, 10, 5, 4, 5]

class CentralTendency:
    def __init__(self, Weights, Frequencies):
        # Convert lists to NumPy arrays for efficient math operations
        self.Wg = np.array(Weights)    # Holds intervals like [(65, 84), ...]
        self.Fq = np.array(Frequencies) # Holds counts like [9, 10, ...]
        # Midpoints: Average of each interval’s lower and upper bounds (e.g., (65 + 84) / 2 = 74.5)
        self.midpoint = (self.Wg[:, 0] + self.Wg[:, 1]) / 2
        # Check if data is continuous: Does the upper bound of one class equal the lower bound of the next at the mode?
        # Example: If mode is at index 2 (105-124), check if 104 == 105 (false here)
        self.is_continuous = np.argmax(self.Fq) > 0 and self.Wg[np.argmax(self.Fq) - 1][1] == self.Wg[np.argmax(self.Fq)][0]
        # Note: We’ll calculate class width (w) dynamically in each method instead of here

    def ArithmeticMean(self):
        # Arithmetic Mean (x̄) = Σ(fi * xi) / Σfi
        # Multiply each midpoint by its frequency, sum them, divide by total frequency
        # Example: (74.5 * 9 + 94.5 * 10 + ...) / 60
        return sum(self.midpoint * self.Fq) / sum(self.Fq)

    def GeometricMean(self):
        # Geometric Mean = exp(Σ(fi * ln(xi)) / Σfi)
        # Take natural log of midpoints, weight by frequencies, average, then exponentiate
        # Useful for growth rates or skewed data
        return np.exp(sum(np.log(self.midpoint) * self.Fq) / sum(self.Fq))

    def HarmonicMean(self):
        # Harmonic Mean = Σfi / Σ(fi / xi)
        # Sum frequencies divided by midpoints, then divide total frequency by that sum
        # Good for rates (e.g., speed)
        return sum(self.Fq) / sum(self.Fq / self.midpoint)

    def Mode(self):
        # Mode: Most frequent class’s value, adjusted by interpolation
        modal_index = np.argmax(self.Fq)  # Find class with highest frequency (e.g., 17 at index 2)
        Lmo = self.Wg[modal_index][0]     # Lower bound of modal class (e.g., 105)
        # Class width: Upper - Lower + 1 if discrete (e.g., 124 - 105 + 1 = 20), else just Upper - Lower
        w = self.Wg[modal_index][1] - self.Wg[modal_index][0] + (1 if not self.is_continuous else 0)
        # Δ1: Difference from previous frequency (or freq if first class)
        Delta1 = self.Fq[modal_index] if modal_index == 0 else self.Fq[modal_index] - self.Fq[modal_index - 1]
        # Δ2: Difference from next frequency (or freq if last class)
        Delta2 = self.Fq[modal_index] if modal_index == len(self.Fq) - 1 else self.Fq[modal_index] - self.Fq[modal_index + 1]
        # Formula: Lmo + w * (Δ1 / (Δ1 + Δ2)) shifts mode within the class
        return Lmo + w * (Delta1 / (Delta1 + Delta2))

    def Median(self):
        # Median: Middle value when data is ordered
        median_pos = sum(self.Fq) / 2  # Position of median (e.g., 60 / 2 = 30)
        # Find class where cumulative frequency reaches or exceeds median position
        median_index = np.argmax(np.cumsum(self.Fq) >= median_pos)  # e.g., cumsum = [9, 19, 36, ...], index 2
        Lmed = self.Wg[median_index][0]  # Lower bound (e.g., 105)
        Fmed = self.Fq[median_index]     # Frequency of median class (e.g., 17)
        Cf = sum(self.Fq[:median_index]) if median_index > 0 else 0  # Cumulative frequency before (e.g., 19)
        # Width specific to this class (not modal width)
        w = self.Wg[median_index][1] - self.Wg[median_index][0] + (1 if not self.is_continuous else 0)
        # Formula: Lmed + (w / Fmed) * (median_pos - Cf) interpolates within the class
        return Lmed + (w / Fmed) * (median_pos - Cf)

    def Quartiles(self, i):
        # Quartiles: Q1 (25%), Q2 (50%), Q3 (75%)
        # Pass i (1, 2, or 3) instead of input for flexibility
        if not (1 <= i <= 3):
            raise ValueError("Quartile must be 1, 2, or 3")
        quartile_pos = (i / 4) * sum(self.Fq)  # e.g., Q1 = 15, Q2 = 30, Q3 = 45
        quartile_index = np.argmax(np.cumsum(self.Fq) >= quartile_pos)
        Lq = self.Wg[quartile_index][0]  # Lower bound of quartile class
        Fq = self.Fq[quartile_index]     # Frequency of quartile class
        Cf = sum(self.Fq[:quartile_index]) if quartile_index > 0 else 0
        w = self.Wg[quartile_index][1] - self.Wg[quartile_index][0] + (1 if not self.is_continuous else 0)
        # Return numeric value for reuse in calculations
        return Lq + (w / Fq) * (quartile_pos - Cf)

    def Deciles(self, i):
        # Deciles: D1 (10%), D2 (20%), ..., D9 (90%)
        if not (1 <= i <= 9):
            raise ValueError("Decile must be 1 to 9")
        decile_pos = (i / 10) * sum(self.Fq)  # e.g., D1 = 6, D5 = 30, D9 = 54
        decile_index = np.argmax(np.cumsum(self.Fq) >= decile_pos)
        Lq = self.Wg[decile_index][0]
        Fq = self.Fq[decile_index]
        Cf = sum(self.Fq[:decile_index]) if decile_index > 0 else 0
        w = self.Wg[decile_index][1] - self.Wg[decile_index][0] + (1 if not self.is_continuous else 0)
        return Lq + (w / Fq) * (decile_pos - Cf)

    def Percentiles(self, i):
        # Percentiles: P1 (1%), P2 (2%), ..., P99 (99%)
        if not (1 <= i <= 99):
            raise ValueError("Percentile must be 1 to 99")
        percentile_pos = (i / 100) * sum(self.Fq)  # e.g., P25 = 15, P50 = 30, P75 = 45
        percentile_index = np.argmax(np.cumsum(self.Fq) >= percentile_pos)
        Lq = self.Wg[percentile_index][0]
        Fq = self.Fq[percentile_index]
        Cf = sum(self.Fq[:percentile_index]) if percentile_index > 0 else 0
        w = self.Wg[percentile_index][1] - self.Wg[percentile_index][0] + (1 if not self.is_continuous else 0)
        return Lq + (w / Fq) * (percentile_pos - Cf)

# Test the class
if __name__ == "__main__":
    Ct = CentralTendency(Weights, Frequencies)
    print("Arithmetic Mean:", Ct.ArithmeticMean())  # Expected: ~122.5
    print("Geometric Mean:", Ct.GeometricMean())    # Expected: ~116.8
    print("Harmonic Mean:", Ct.HarmonicMean())      # Expected: ~110.7
    print("Mode:", Ct.Mode())                       # Expected: 115
    print("Median:", Ct.Median())                   # Expected: ~117.941
    print("Quartiles:", Ct.Quartiles(1), Ct.Quartiles(2), Ct.Quartiles(3))  # Expected: 97, ~117.941, 143
    print("Deciles (D5):", Ct.Deciles(5))           # Expected: ~117.941
    print("Percentiles (P50):", Ct.Percentiles(50)) # Expected: ~117.941