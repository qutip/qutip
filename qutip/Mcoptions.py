
class Mcoptions():
    def __init__(self):
        """
        Class of options for Monte-Carlo ODE solver (mcsolve.py)
        """
        self.atol=1e-12
        self.rtol=1e-6
        self.method='adams'
        self.nsteps=1000
        self.first_step=0
        self.min_step=0
        self.max_step=0
        self.order=12
        self.progressbar=True
    def __str__(self):
        print "OPTIONS FOR MONTE-CARLO ODE SOLVER:"
        print "------------------------------------"
        print 'atol =',self.atol,': Absolute tolerance (default = 1e-12)\n'
        print 'rtol =',self.rtol,': Relative tolerance (default = 1e-6)\n'
        print 'method =',self.method,': Integration method (default="adams", use "bdf" for stiff problems)\n'
        print 'nsteps =',self.nsteps,': Max. number of internal steps allowed per call (default=1000)\n'
        print 'first_step =',self.first_step,': Size of initial step (default=0, determined by solver)\n'
        print 'min_step =',self.min_step,': Minimal step size (default=0, determined by solver)\n'
        print 'max_step =',self.max_step,': Max step size (default=0, determined by solver)\n'
        print 'order =',self.order,': Maximum order used by integrator (<=12 for "adams", <=5 for "bdf")'
        return ''


if __name__ == "__main__":
    print Mcoptions()
