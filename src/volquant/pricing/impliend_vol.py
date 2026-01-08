import warnings
from volquant.models import black_scholes
import numpy as np
import math
from enum import Enum
from scipy.optimize import brentq

class SolverMethod(Enum):
    NEWTON = "newton"
    BISECTION = "bisection"
    BRENT = "brent"
def validate_price(S : float , K : float , r : float , T : float , option_type : black_scholes.OptionType,q:float,price : float)-> None:
    if option_type == black_scholes.OptionType.CALL:
        intrinsic_value = max(S*np.exp(-q*T) - K*np.exp(-r*T),0.0)
        max_value = S*np.exp(-q*T)
    else:
        intrinsic_value = max(K*np.exp(-r*T) - S*np.exp(-q*T),0.0)
        max_value = K*np.exp(-r*T)

    if price < intrinsic_value or price > max_value:
        raise ValueError(f"The price {price} has an arbitrage opportunity")


def initial_guess_bs(S : float , T : float , price : float) -> float:
    return np.sqrt(2*np.pi/T)*(price/S)

def initial_guess_mannaster(S : float , K : float , r : float , T : float):
    return np.sqrt(2*abs(np.log(S/K)+r*T)/T)

def initial_guess(S :float , K : float , r : float , T : float , price : float):
    guesses = []
    sigma_bs = initial_guess_bs(S , T  , price )
    if 0.01 < sigma_bs < 3.0:
        guesses.append(sigma_bs)
    sigma_mk = initial_guess_mannaster(S ,K , r , T)
    if 0.01 < sigma_mk < 3.0:
        guesses.append(sigma_mk)
    guesses.append(0.2)

    return np.median(guesses)

def implied_vol_newton(S : float , K : float , r : float , T : float,option_type : black_scholes.OptionType, price : float , q : float = 0.0,tolerance : float = 1e-8,max_iterations :int = 50,sigma_init : float = None) -> float:
    validate_price(S , K , r , T , option_type , q , price)
    if sigma_init is None:
        vol = initial_guess(S,K,r,T,price)

    else:
        vol = sigma_init

    iterations = 1

    while iterations <= max_iterations:
        diff = black_scholes.Blackscholes_price(S,K,r,T,vol,option_type,q) - price
        if abs(diff) < tolerance:
            return vol

        vega = black_scholes.vega(S,K,r,T,vol,q)
        if not math.isfinite(vega) or abs(vega) < 1e-8:
            raise ValueError("Newtons method will not converge as vega is too small")

        vol = vol - (diff/vega)

        if not math.isfinite(vol):
            raise ValueError("Newton method failed : vol became non finite")

        vol = min(max(0.01,vol),3.0)

        iterations += 1

    raise ValueError(f"The Newtons Method didnot converge in {iterations-1} iterations")


def implied_vol_bisection(S : float , K : float , r : float , T : float , option_type : black_scholes.OptionType,price : float, q : float = 0.0, tolerance : float = 1e-8 , max_iterations : int = 50,sigma_low : float = 0.01,sigma_high : float =5.0 )->float:

    validate_price(S , K , r , T , option_type , q , price)
    if sigma_low <= 0 or sigma_high <= 0 or sigma_low >= sigma_high:
        raise ValueError(f"Invalid initial volatility bracket [{sigma_low}, {sigma_high}]")

    f_low = black_scholes.Blackscholes_price(S,K,r , T , sigma_low ,option_type,q)-price
    f_high = black_scholes.Blackscholes_price(S,K,r,T,sigma_high,option_type,q)-price 

    if f_low*f_high >  0:
        raise ValueError(f"The given value range [{sigma_low},{sigma_high} doesnt contain the root")

    for i in range(max_iterations):
        sigma_mid = (sigma_low + sigma_high)/2
        f_mid = black_scholes.Blackscholes_price(S, K , r , T ,sigma_mid,option_type,q)-price

        if abs(f_mid) < tolerance:
            return sigma_mid 

        if f_mid*f_low < 0:
            sigma_high = sigma_mid
            f_high = f_mid

        else:
            sigma_low = sigma_mid
            f_low = f_mid

    raise ValueError(f"The bisection method didnot converge in {max_iterations} iteraions")

def implied_vol_brent(S : float , K : float , r : float , T : float , option_type : black_scholes.OptionType,price : float, q : float = 0.0, tolerance : float = 1e-8,sigma_low = 1e-4,sigma_high=5.0 )->float:
    validate_price(S,K,r,T, option_type,q , price)

    def objective(sigma):
        return black_scholes.Blackscholes_price(S,K,r,T,sigma,option_type,q) - price

    try:
        return brentq(objective,sigma_low,sigma_high,xtol = tolerance)

    except Exception as e:
        raise ValueError("Brents Method failed") from e





def implied_vol(S : float , K : float , r : float , T : float , option_type : black_scholes.OptionType,price : float, q : float = 0.0, tolerance : float = 1e-8 , max_iterations : int = 50,method:str|SolverMethod = SolverMethod.BRENT)->float:
    if isinstance(method,str):
        method = SolverMethod(method.lower())

    if method == SolverMethod.NEWTON:
        return implied_vol_newton(S,K,r,T,option_type,price,q,tolerance,max_iterations)

    if method == SolverMethod.BISECTION:
        return implied_vol_bisection(S,K,r ,T , option_type , price , q , tolerance , max_iterations)

    if method == SolverMethod.BRENT:
        return implied_vol_brent(S , K , r , T , option_type , price , q ,tolerance)



    






