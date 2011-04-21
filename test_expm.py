from scipy import *
from scipy.linalg import norm,solve


def test_expm(A):
    #############################
    def pade(m):
        n=len(A)
        c=padecoeff(m)
        if m!=13:
            apows= [[] for jj in xrange(int(ceil((m+1)/2)))]
            apows[0]=eye(n,n)
            apows[1]=dot(A,A)
            for jj in xrange(2,int(ceil((m+1)/2))):
                apows[jj]=dot(apows[jj-1],apows[1])
            U=zeros((n,n)); V=zeros((n,n))
            for jj in xrange(m,0,-2):
                U+=c[jj]*apows[jj/2]
            U=dot(A,U)
            for jj in xrange(m-1,-1,-2):
                V+=c[jj]*apows[(jj+1)/2]
            F=solve((-U+V),(U+V))
            return F
        elif m==13:
            A2=dot(A,A)
            A4=dot(A2,A2)
            A6=dot(A2,A4)
            U = dot(A,(dot(A6,(c[13]*A6+c[11]*A4+c[9]*A2))+c[7]*A6+c[5]*A4+c[3]*A2+c[1]*eye(n,n)))
            V = dot(A6,(c[12]*A6 + c[10]*A4 + c[8]*A2))+ c[6]*A6 + c[4]*A4 + c[2]*A2 + c[0]*eye(n,n)
            F=solve((-U+V),(U+V))
            return F
    #################################
    m_vals=array([3,5,7,9,13])
    theta=array([0.01495585217958292,0.2539398330063230,0.9504178996162932,2.097847961257068,5.371920351148152],dtype=float)
    normA=norm(A,1)
    if normA<=theta[-1]:
        for ii in xrange(len(m_vals)):
            if normA<=theta[ii]:
                F=pade(m_vals[ii])
                break
    else:
        t,s=frexp(normA/theta[-1])
        s=s-(t==0.5)
        A=A/2.0**s
        F=pade(m_vals[-1])
        for i in xrange(s):
            F=dot(F,F)
    return F
        

def padecoeff(m):
    if m==3:
        return array([120, 60, 12, 1])
    elif m==5:
        return array([30240, 15120, 3360, 420, 30, 1])
    elif m==7:
        return array([17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1])
    elif m==9:
        return array([17643225600, 8821612800, 2075673600, 302702400, 30270240,2162160, 110880, 3960, 90, 1])
    elif m==13:
        return array([64764752532480000, 32382376266240000, 7771770303897600,1187353796428800, 129060195264000, 10559470521600,670442572800, 33522128640, 1323241920,40840800, 960960, 16380, 182, 1])















