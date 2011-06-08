
class Mcoptions():
    def __init__(self):
        """
        Class of options for Monte-Carlo ODE solver mcsolve.
        
        Class attributes:
        -----------------
        atol:       Absolute tolerance (default = 1e-12)
        rtol:       Relative tolerance (default = 1e-8)
        method:     Integration method (default = 'adams')
        steps:      Max. number of internal steps/call
        first_step: Size of initial step (determined by solver)
        min_step:   Minimal step size (determined by solver)
        max_step:   Max step size (determined by solver)
        order:      Maximum order used by integrator (12)
        states_out: Return kets instead of expect. values (False).
        
        """
        self.atol=1e-12
        self.rtol=1e-8
        self.method='adams'
        self.nsteps=1000
        self.first_step=0
        self.min_step=0
        self.max_step=0
        self.order=12
        self.progressbar=True
        self.states_out=False
    def __str__(self):
        print "OPTIONS FOR MONTE-CARLO ODE SOLVER:"
        print "------------------------------------"
        print 'atol =',self.atol,': Absolute tolerance (default = 1e-12)\n'
        print 'rtol =',self.rtol,': Relative tolerance (default = 1e-8)\n'
        print 'method =',self.method,': Integration method (default="adams", use "bdf" for stiff problems)\n'
        print 'nsteps =',self.nsteps,': Max. number of internal steps allowed per call (default=1000)\n'
        print 'first_step =',self.first_step,': Size of initial step (default=0, determined by solver)\n'
        print 'min_step =',self.min_step,': Minimal step size (default=0, determined by solver)\n'
        print 'max_step =',self.max_step,': Max step size (default=0, determined by solver)\n'
        print 'order =',self.order,': Maximum order used by integrator (<=12 for "adams", <=5 for "bdf")\n'
        print 'states_out =',self.states_out,': Return kets instead of expect. values. (default=False)'
        return ''


if __name__ == "__main__":
    print Mcoptions()
