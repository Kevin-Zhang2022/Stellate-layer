import torch.nn as nn
import torch
import numpy as np

from numpy import (arange, array, pi, cos, exp, log10, ones_like, sqrt, zeros)
try:
    from scipy.misc import factorial
except ImportError:
    from scipy.special import factorial


_ERB_L = 24.7
_ERB_Q = 9.265

def erb_count(centerfrequency):
    """Returns the equivalent rectangular band count up to centerfrequency.

    Parameters
    ----------
    centerfrequency : scalar /Hz
        The center frequency in Hertz of the
        desired auditory filter.

    Returns
    -------
    count : scalar
        Number of equivalent bandwidths below `centerfrequency`.

    """
    return 21.4 * log10(4.37 * 0.001 * centerfrequency + 1)


def erb_aud(hz):
    """Retrurns equivalent rectangular band width of an auditory filter.
    Implements Equation 13 in [Hohmann2002]_.

    Parameters
    ----------
    centerfrequency : scalar /Hz
        The center frequency in Hertz of the
        desired auditory filter.

    Returns
    -------
    erb : scalar
        Equivalent rectangular bandwidth of
        an auditory filter at `centerfrequency`.

    """
    return _ERB_L + hz / _ERB_Q


def hertz_to_erbscale(frequency):
    """Returns ERB-frequency from frequency in Hz.
    Implements Equation 16 in [Hohmann2002]_.

    Parameters
    ----------
    frequency : scalar
        The Frequency in Hertz.

    Returns
    -------
    erb : scalar
        The corresponding value on the ERB-Scale.

    """
    return 21.4 * np.log10(1 + frequency / (_ERB_L * _ERB_Q))


def erbscale_to_hertz(erb):
    """Returns frequency in Hertz from ERB value.
    Implements Equation 17 in [Hohmann2002]_.

    Parameters
    ----------
    erb : scalar
        The corresponding value on the ERB-Scale.

    Returns
    -------
    frequency : scalar
        The Frequency in Hertz.

    """
    return (10**(erb/21.4) - 1) * _ERB_L * _ERB_Q

class GTFB(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            fr_hz=np.array([20,20000]),  # fr frequency range
            bf=0.05,
            fs=441000,
    ):
        super(GTFB, self).__init__()
        self.out_features=out_features
        self.bf=bf
        self.fs = fs
        self.in_features=in_features
        self.fr_hz=np.array(fr_hz)
        self.fr_erb = hertz_to_erbscale(self.fr_hz)
        self.cf_erb = np.arange(self.fr_erb[0],self.fr_erb[1],(self.fr_erb[1]-self.fr_erb[0])/self.out_features)[0:out_features]

        self.cf_hz = erbscale_to_hertz(self.cf_erb)
        self.bw = erb_aud(self.cf_hz)*self.bf
        phi = pi * self.bw / self.fs
        alpha = 10**(0.1 * -3/4)
        p = (-2 + 2 * alpha * cos(phi)) / (1 - alpha)
        lambda_ = -p/2 - sqrt(p*p/4 - 1)
        beta = 2*pi * self.cf_hz / self.fs
        self.a = lambda_ * exp(1j*beta)
        self.nf = torch.tensor(2 * (1 - abs(self.a))**4)
        self.a1 = torch.tensor(4 * self.a)
        self.a2 = torch.tensor(-6 * pow(self.a, 2))
        self.a3 = torch.tensor(4 * pow(self.a, 3))
        self.a4 = torch.tensor(-pow(self.a, 4))
        self.yn1 = torch.tensor(0)
        self.yn2 = torch.tensor(0)
        self.yn3 = torch.tensor(0)
        self.yn4 = torch.tensor(0)
        # b, a = array([factor]), array([1., -coef])

    def reset(self):
        self.yn1 = torch.tensor(0)
        self.yn2 = torch.tensor(0)
        self.yn3 = torch.tensor(0)
        self.yn4 = torch.tensor(0)

    def forward(self,x):
        x = x.unsqueeze(1)
        yn = x * self.nf + self.a1 * self.yn1 + self.a2 * self.yn2 + self.a3 * self.yn3 + self.a4 * self.yn4
        self.yn4 = self.yn3
        self.yn3=self.yn2
        self.yn2 = self.yn1
        self.yn1 = yn
        return torch.real(yn)





