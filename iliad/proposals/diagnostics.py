import numpy as np


class Diagnostics:
    """Convenience class for working with averages and summations of diagnostic
    quantities. Also keeps a list of all recorded values.

    Parameters:
        values: List of values.
        avg: The average of the list of valus.
        summ: The sum of the list of values.
        num_values: The number of values used in computing the sum or average.

    """
    def __init__(self):
        self.values: list = []
        self.summ: float = np.nan
        self.num_values: int = 0

    @property
    def avg(self):
        if self.num_values > 0:
            return self.summ / self.num_values
        else:
            return np.nan

    def update(self, val: float):
        self.values.append(val)
        if not np.isnan(val) and not np.isinf(val):
            self.num_values += 1
            if self.num_values == 1:
                self.summ = val
            else:
                self.summ += val
