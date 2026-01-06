import numpy as np
from scipy.stats import norm
import math
from enum import Enum

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

def validate_inputs(S : float , K : float , T : float , sigma : float) -> None:
    #check whether the provided inputs are in valid range
    if S <= 0 :
        raise ValueError(f"The value of stock price cannot be negative, got {S}")
    if K <= 0 :
        raise ValueError(f"Strike price cannot be negative, got {K}")
    if T < 0 :
        raise ValueError(f"Time to maturity cannot be negative, got {T}")
    if T > 0 and sigma <= 0:
        raise ValueError(f"The volatility of an should be positive when T > 0 , got {sigma}"
        )

def d1_d2(S : float , K : float , r : float , T : float , sigma : float , q : float ) -> tuple:
    if T == 0:
        if S > K:
            return math.inf , math.inf
        elif S < K :
            return -math.inf , -math.inf
        return math.nan , math.nan
    d1 = (np.log(S/K) + (r-q+ 0.5*sigma*sigma)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    return d1 , d2

def Blackscholes_price(S : float , K : float , r : float , T : float , sigma : float ,option_type: str | OptionType, q : float = 0.0) -> float :
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
    if isinstance(option_type,str):
        option_type = OptionType(option_type.lower())

    if T == 0:
        if option_type == OptionType.CALL:
            return max(S-K , 0)
        else:
            return max(K-S,0)

    d1 , d2 = d1_d2(S , K , r ,T , sigma , q)
    if option_type == OptionType.CALL:
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def parity_check(call_price : float , put_price : float , S : float , K : float , r: float, T : float , q : float = 0.0 , tolerance : float = 1e-6) -> dict:

    #check whether the market call and put prices holds put call parity
    lhs = call_price - put_price
    rhs = S * np.exp(-q*T) - K * np.exp(-r*T)
    parity_holds = (abs(lhs-rhs) <= tolerance)

    return {
        "tolerance_level" : tolerance,
        "parity_holds" : parity_holds
    }


def delta(S : float , K : float , r : float , T : float , sigma : float , option_type : str| OptionType = OptionType.CALL, q : float = 0.0) -> float :
    validate_inputs(S,K,T,sigma)
    if isinstance(option_type,str):
        option_type = OptionType(option_type.lower())

    if T == 0:
        if option_type == OptionType.CALL:
            if S == K:
                return 0.5
            elif S > K:
                return 1.0
            else:
                return 0.0

        else:
            if S == K:
                return -0.5
            elif S < K:
                return -1.0
            else:
                return 0.0

    d1 , _ = d1_d2(S , K , r , T , sigma , q)
    if option_type == OptionType.CALL:
        return np.exp(-q*T)*norm.cdf(d1)

    else:
        return np.exp(-q*T)*norm.cdf(-d1)

def gamma(S : float , K : float , r : float , T : float , sigma : float , q : float = 0.0)-> float:
    validate_inputs(S,K,T,sigma)

    if T == 0:
        return math.inf if S == K else 0.0

    d1 , _ = d1_d2(S , K , r , T , sigma , q)

    return np.exp(-q*T)*norm.pdf(d1)/(S*sigma*np.sqrt(T))

def vega(S : float , K : float , r : float , T : float , sigma : float , q : float = 0.0) -> float:
    validate_inputs(S,K,T,sigma)

    if T == 0 :
        return 0.0

    d1 , _ = d1_d2(S,K,r,T,sigma,q)

    return np.exp(-q*T)* norm.pdf(d1)*S*np.sqrt(T)

def rho(S: float , K : float , r: float , T : float , sigma : float ,option_type : str|OptionType = OptionType.CALL, q : float = 0.0)-> float:
    validate_inputs(S,K,T,sigma)

    if isinstance(option_type ,str):
        option_type = OptionType(option_type.lower())

    if T == 0:
        return 0.0

    _, d2 = d1_d2(S , K , r , T , sigma , q)

    if option_type == OptionType.CALL:
        return K*T*np.exp(-r*T)*norm.cdf(d2)

    else:
        return -K* T*np.exp(-r*T)*norm.cdf(-d2)

def theta(S : float , K : float , r : float , T : float , sigma : float , option_type : str | OptionType = OptionType.CALL,q:float = 0.0 ) -> float:
    validate_inputs(S,K,T,sigma)

    if isinstance(option_type,str):
        option_type = OptionType(option_type.lower())


    if T == 0:
        return -math.inf if S == K else 0.0

    d1 , d2 = d1_d2(S , K , r , T , sigma , q)

    term1 = -S*np.exp(-q*T)*norm.pdf(d1)*sigma/(2*np.sqrt(T))

    if option_type == OptionType.CALL:
        term2 = q*S*np.exp(-q*T)*norm.cdf(d1)
        term3 = -r*K*np.exp(-r*T)*norm.cdf(d2)
        return term1 + term2 + term3

    else:
        term2 = -q*S*np.exp(-q*T)*norm.cdf(-d1)
        term3 = r*K*np.exp(-r*T)*norm.cdf(-d2)
        return term1 + term2 + term3



def all_greeks(S : float , K : float , r : float , T : float , sigma : float , option_type : str |OptionType = OptionType.CALL,q:float = 0.0)->dict:
    return {
        "delta" : delta(S,K,r,T,sigma , option_type , q),
        "gamma" : gamma(S,K,r,T,sigma,q),
        "vega" : vega(S,K,r,T,sigma,q),
        "rho" : rho(S,K,r,T,sigma,option_type,q),
        "theta" : theta(S,K,r,T,sigma,option_type,q)
    }
    




    






