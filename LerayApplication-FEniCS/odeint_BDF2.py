
class odeint_BDF2():

    def __init__(self, **kwargs):
        r"""Initialises the ODE integrator (single-step-method).
            Uses underneath the generalised alpha method and its limits:

            BDF2:
            alpha_0 = 1.5, alpha_1 = -2.0, alpha_2 = 0.5
        Args:
            **kwargs:
                dt: Time step size.
                alpha_0
                alpha_1
                alpha_2
        """
        # Eval settings

        # Time step size
        if 'dt' in kwargs:
            self.dt = kwargs['dt']
        else:
            raise ArgumentError("No time step dt given.")

        # Default parameters 
        self.alpha_0 = 1.5
        self.alpha_1 = -2.0
        self.alpha_2 = 0.5

        
        # pointers to solution states (not documented)
        if 'x' in kwargs:
            self.x = kwargs['x']
        if 'x0' in kwargs:
            self.x_last_time = kwargs['x0']
        if 'x00' in kwargs:
            self.x_last_last_time = kwargs['x00']

        if 'verbose' in kwargs and kwargs['verbose'] is True:
            out = "alpha_0 = %.3f, alpha_1 = %.3f, alpha_2 = %.3f" % \
                  (self.alpha_0, self.alpha_1, self.alpha_2)
            print("odeint_BDF2 using: %s" % out )

    def g_(self, g, x=None, x0=None, x00=None):
        if g is None: raise ArgumentError("No function or expression given.")
        if x is None: x=self.x
        if x0 is None: x0=self.x_last_time
        if x00 is None: x00=self.x_last_last_time

        # TODO: Rethink wrt non-constant expressions involving time derivative

        g_x = g(x, time_instant=1)
        g_x0 = g(x0, time_instant=0)
        g_x00 = g(x00, time_instant=-1)

        # local function to compute the expression
        def _compute(g_x, g_x0, g_x00):
            g_xt = self.alpha_0 * g_x + self.alpha_1 * g_x0 + self.alpha_2 * g_x00
            return g_xt

        if type(g_x) is list and type(g_x0) is list and type(g_x00) is list:
            # check dimensions
            assert(len(g_x) == len(g_x0))
            assert(len(g_x) == len(g_x00))
            # return list of forms version
            return [ _compute(_g_x, _g_x0, _g_x00) for _g_x, _g_x0, _g_x00 in zip(g_x, g_x0, g_x00) ]
        else:
            # return form version
            return _compute(g_x, g_x0, g_x00)

  

    def update(self, x=None, x0=None, x00=None):
        if x is None: x=self.x
        if x0 is None: x0=self.x_last_time
        if x00 is None: x00=self.x_last_last_time

        x0.assign( x )
        x00.assign( x0)

        return x0, x00
