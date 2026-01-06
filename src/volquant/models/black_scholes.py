import numpy as np
from scipy.stats import norm
import math
from enum import Enum

def validate_inputs(S : float , K : float , T : float , sigma : float) -> None:
    #check whether the provided inputs are in valid range
    if S <= 0 :
        raise ValueError(f"The value of stock price cannot be negative, got {S}")
    if K <= 0 :
        raise ValueError(f"Strike price cannot be negative, got {K}")
    if T < 0 :
        raise ValueError(f"Time to maturity cannot be negative, got {T}")
    if sigma <= 0:
        raise ValueError(f"The volatility of an should be positive , got {sigma}"
        )

def d1_d2(S : float , K : float , r : float , T : float , sigma : float , q : float ) -> tuple:
    if T == 0:
        if S > K:
            return math.inf , math.inf
        elif S < K :
            return -math.inf , -math.inf
        else:
            return 0.0 , 0.0
    d1 = (np.log(S/K) + (r-q+ 0.5*sigma*sigma)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    return d1 , d2

def Blackscholes_price(S : float , K : float , r : float , T : float , sigma : float ,option_type: , q : float = 0.0) -> float :
    """
    The parameters are :
    S : Stock price 
    K : Strike price
    r : risk free interest rate(continuous)
    T : Time to maturity entered in years
    sigma : volatility of the underlying (annualized)
    q : continuous dividend rate 
    """

    validate_inputs(S,K,T,sigma)

    d1 , d2 = d1_d2(S , K , r ,T , sigma , q)






