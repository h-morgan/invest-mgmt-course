import pandas as pd
import scipy
import numpy as np
from scipy.stats import norm

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns 
    Computes and returns a DataFrame that contains: the wealth index, the previous peaks, and percent drawdowns 
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })


def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv(r"/Users/haleymorgan/Documents/misc./personal/QuantDevResources/InvestmentManagementCourse/data/Portfolios_Formed_on_ME_monthly_EW.csv",
                     header=0, index_col=0, parse_dates=True, na_values=-99.99
                      )
    # Grab the columns of data we want to look at (the funds we are interested in analyzing)
    returns = me_m[["Lo 10", "Hi 10"]]
    # Rename the columns 
    returns.columns = ['SmallCap', 'LargeCap']
    # Divide values by 100 to convert from percent values to ints 
    returns = returns/100
    returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period("M")
    return returns


def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv(r"/Users/haleymorgan/Documents/misc./personal/QuantDevResources/InvestmentManagementCourse/data/edhec-hedgefundindices.csv",
                     header=0, index_col=0, parse_dates=True
                      )
    # Divide values by 100 to convert from percent values to ints, format index col to be dates
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi


def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied series or DataFrame
    Returns a float or a series
    """
    demeaned_r = r - r.mean()
    # Use the population standard deviation, so set degrees of freedom dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied series or DataFrame
    Returns a float or a series
    """
    demeaned_r = r - r.mean()
    # Use the population standard deviation, so set degrees of freedom dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default (1% level of confidence at least, same as saying p value should be 1%)
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level 


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. Returns the number such that the "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level) # this calls "some function" over every column of this return series - the some function in this case is var_historic
        # This now passes a series of aggregate columns to var_historic fn, and is now handled
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")
        

def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    # Compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewnewss and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z + 
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        # Find all of the returns that are less than the historic VaR
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or a DataFrame")