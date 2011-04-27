import sys
import dis
def varargout():
    """
    Returns the number of outputs requested by the user.
    Acts like Matlab varargout function. 
    """
    f = sys._getframe().f_back.f_back
    i = f.f_lasti + 3
    bytecode = f.f_code.co_code
    instruction = ord(bytecode[i])
    while True:
        if instruction == dis.opmap['DUP_TOP']:
            if ord(bytecode[i + 1]) == dis.opmap['UNPACK_SEQUENCE']:
                return ord(bytecode[i + 2])
            i += 4
            instruction = ord(bytecode[i])
            continue
        if instruction == dis.opmap['STORE_NAME']:
            return 1
        if instruction == dis.opmap['UNPACK_SEQUENCE']:
            return ord(bytecode[i + 1])
        return 0



if __name__ == "__main__":
    def example():
        r = varargout()
        if r==0:
            print 'none'
            return 0
        if r==1:
            print 'one'
            return 1
        if r==2:
            print 'two'
            return 1,2
    
    example()
    a=example()
    a,b=example()


            
