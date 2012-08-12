#function to run the nose test scripts
def run():
    import nose
    nose.run(defaultTest="qutip.tests",argv=['nosetests', '-v']) #runs tests in qutip.tests module only

