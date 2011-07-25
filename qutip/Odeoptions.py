##
#Class of options for ODE solvers.
#
class Odeoptions():
    def __init__(self):
        ##Absolute tolerance (default = 1e-12)
        self.atol=1e-10
        ##Relative tolerance (default = 1e-8)
        self.rtol=1e-8
        ##Integration method (default = 'adams', for stiff 'bdf')
        self.method='adams'
        ##Max. number of internal steps/call
        self.nsteps=1000
        ##Size of initial step (0 = determined by solver)
        self.first_step=0
        ##Minimal step size (0 = determined by solver)
        self.min_step=0
        ##Max step size (0 = determined by solver)
        self.max_step=0
        ##Maximum order used by integrator (<=12 for 'adams', <=5 for 'bdf')
        self.order=12
    def __str__(self):
        print "Odeoptions properties:"
        print "----------------------"
        print 'atol:       ',self.atol
        print 'rtol:       ',self.rtol
        print 'method:     ',self.method
        print 'order:      ',self.order
        print 'nsteps:     ',self.nsteps
        print 'first_step: ',self.first_step
        print 'min_step:   ',self.min_step
        print 'max_step:   ',self.max_step
        return ''


if __name__ == "__main__":
    print Odeoptions()
