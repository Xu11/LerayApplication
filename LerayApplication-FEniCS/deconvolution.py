
class deconvolution():

    def __init__(self, **kwargs):
        r"""Initialises the ODE integrator (single-step-method).
            Uses underneath the generalised alpha method and its limits:

            Euler forward:
                alpha_f = 0, alpha_m = 1/2, gamma = 1/2
            Euler backward:
                alpha_f = 1, alpha_m = 1/2, gamma = 1/2
            Crank-Nicolson:
                alpha_f = 1/2, alpha_m = 1/2, gamma = 1/2
            Theta:
                alpha_f = theta, alpha_m = 1/2, gamma = 1/2
            Generalised alpha:
                The value of rho can be used to determine the values
                alpha_f = 1/(1+rho),
                alpha_m = 1/2*(3-rho)/(1+rho),
                gamma = 1/2 + alpha_m - alpha_f
        Args:
            **kwargs:
                dt: Time step size.
                rho: Spectral radius rho_infinity for generalised alpha.
                alpha_f:
                alpha_m:
                gamma:
        """
        # Eval settings

        self.N = kwargs['N']
        self.delta = kwargs['delta']
        self.vel = kwargs['velocity']

    def applyhelmotzfilter(self,phi,datum='Neumann'):
        # that is I-F_H

        # Define helmotz equation


        # Define Null Space or Dirichlet Datum.


        # Solve Helmotz equation


    def compute_indicator(self):
        # with Yoshida regolarization
        # a_N(u) = [delta^2N+2 | ([delta^2(I-F_H)]^(N+1) u)
        
        
