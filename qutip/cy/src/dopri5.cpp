#include <complex>
#include <vector>
#include <iostream>
#include <cmath>

typedef std::complex<double> cplx;

class spmat{
	public:
		cplx *data;
		int size;
		int *ind, *ptr;
		spmat() {};
		spmat(int _size) {size=_size;}; //Python own the data
		~spmat() {};
		void v_mult(cplx *in, cplx *out) const{
		    for( int i=0;i<size;++i){
		    	out[i] = cplx(0.0, 0.0);
		    	for(int j=ptr[i];j<ptr[i+1];++j){
		    		out[i] += data[j]*in[ind[j]];
		    	}
		    }
		};
};

void dxdt(cplx *in, cplx *out, const spmat &H){
	H.v_mult(in, out);
}

double norm2(const cplx *in, const int l){
	double result = 0.0;
	double *dptr=(double*) in;
	for(int i=0;i<l*2;++i){ result += dptr[i]*dptr[i];}
	return result;
}


class ode{
private:
	spmat H;
	int l;
	int norm_step;
	double err[3];
	double atol, rtol, min_step, max_step, norm_tol;

	double dt;
	cplx *derr_in;
	cplx *derr_out;

public:
	ode(){std::cout << "Ode vide\n";};

	ode(int *_int_opt, double *_double_opt, int *_h_ptr, int *_h_ind, cplx *_h_data){
		l = _int_opt[0];
		norm_step = _int_opt[1];
		atol = _double_opt[0];
		rtol = _double_opt[1];
		min_step = _double_opt[2];
		max_step = _double_opt[3];
		if (max_step<=0.) max_step=1e150;
		norm_tol = _double_opt[4];
		H = spmat(l);
		H.ind=_h_ind;
		H.ptr=_h_ptr;
		H.data=_h_data;
		dt = min_step;
		if(min_step ==0.) dt = std::pow(atol,0.25);

		derr_in = new cplx[l];
		derr_out = new cplx[l];

	};
	~ode(){
		delete[] derr_in;
		delete[] derr_out;
	};

	void run(double t_in, double t_final, cplx *psi_in, cplx * psi_out);
	int step(double &, cplx *, cplx *);
	double integrate(double, double, double, cplx *, double*);
	void dopri5(const double &, cplx *, cplx *);
};

void ode::run(double t_in, double t_final, cplx *psi_in, cplx * psi_out){
	double dtt = t_final-t_in;
	dxdt(psi_in, derr_in, H);
	dopri5(dtt, psi_in, psi_out);
}

//Do a dopri step while correcting for the error.
//dt is the approximation of a good step size.
//dtt is the desired/max step size.
//Need to check boost/zvode/scipy for their methods
int ode::step(double &dtt, cplx *psi_in, cplx *psi_out){
	int count=0;

	double ratio = dtt/dt; // Check if the time-step is too big
	if (ratio >2) {
		dtt = dt;
	}

	dopri5(dtt, psi_in, psi_out);
	while(err[2]>=1. && count<2){
		dtt *= std::pow(2*err[2],-0.25);
		dopri5(dtt, psi_in, psi_out);
		count++;
	}
	if (count==2) return 0;

	if (ratio > 0.5){ // New dt if ddt meaningful
		dt = dtt*std::pow(2*err[2],-0.25);
		if(dt>max_step) dt = max_step;
	}
	return 1;
}

double ode::integrate(double t_in, double t_target, double rand, cplx *psi, double *debug){
	double t = t_in;
	double dtt = dt, dt_guess;
	double norm_b, norm_a,norm_guess;
	bool success;
	cplx *psi_out =  new cplx[l];
	int steps =0;
	
	if( dtt > t_target - t ) dtt = t_target - t;
	norm_b = norm2(psi,l);

	debug[0] = t_in;
	debug[1] = t_target;
	debug[2] = dtt;
	debug[4] = 0.;
	debug[5] = norm_b;
	debug[7] = rand;

	while(t<t_target && steps<100){
		dxdt(psi, derr_in, H);
		success = step(dtt, psi, psi_out);
		debug[3] = dtt;
		debug[8] = err[2];
		debug[9] = err[0];
		debug[10] = err[1];

		if(!success){
			delete[] psi_out;
			return -1;
		}
		norm_a = norm2(psi_out,l);
		debug[6] = norm_a;

		if( norm_a <= rand ){ //found a collapse
			int ii=norm_step;
			while(ii--){
				dt_guess = std::log(norm_b/rand)/std::log(norm_b/norm_a)*dtt;
				dopri5(dt_guess, psi, psi_out);

				debug[4] = dt_guess;
				debug[8] = err[2];
				debug[9] = err[0];
				debug[10] = err[1];
				if(err[2]>1.){ 
					delete[] psi_out;
					return -2; 
				}

				norm_guess = norm2(psi_out,l);
				if(std::abs( rand - norm_guess ) < norm_tol*rand ){
					t += dt_guess;
					for(int i=0;i<l;++i) psi[i]=psi_out[i];
					norm_b=norm_guess;
					break;
				}else if(norm_guess<rand){
					norm_a = norm_guess;
					dtt = dt_guess;
				}else{
					norm_b=norm_guess;
					t += dt_guess;
					dtt -= dt_guess;
					for(int i=0;i<l;++i) psi[i]=psi_out[i];
					std::swap(derr_in,derr_out);
				}
			}

			delete[] psi_out;
			if(ii==0){return -3;}
			return t;

		} else {
			t += dtt;
			norm_b = norm_a;
			if( dtt > t_target - t ) dtt = t_target - t;
			for(int i=0;i<l;++i) psi[i]=psi_out[i];
			std::swap(derr_in,derr_out);
		}
		steps++;
	}
	delete[] psi_out;
	if(steps==100)return -4;
	return t;
}

void ode::dopri5(const double &dt, cplx *in, cplx *out){
	cplx *_dxdt0 = derr_in;
	cplx *_dxdt1 = new cplx[l];
	cplx *_dxdt2 = new cplx[l];
	cplx *_dxdt3 = new cplx[l];
	cplx *_dxdt4 = new cplx[l];
	cplx *_dxdt5 = new cplx[l];
	cplx *_dxdt6 = derr_out;
	cplx *_errout = new cplx[l];

	//static const double c[6]  = {(double)1./5.,(double)3./10.,(double)4./5.,(double)8./9., 1., 1.};
	static const double a1[1] = {(double)1./5.};
	static const double a2[2] = {(double)3./40.,   (double)9./40.};
	static const double a3[3] = {(double)44./45.,  (double)-56./15.,(double)32./9.};
	static const double a4[4] = {(double)19372./6561.,(double)-25360./2187.,(double)64448./6561.,
						  (double)-212./729.};
	static const double a5[5] = {(double)9017./3168.,(double)-355./33.,(double)46732./5247.,
						  (double)49./176., (double) -5103./18656.};
	static const double a6[6] = {(double)35./384., 0.,(double)500./1113.,(double)125./192.,
						  (double)-2187./6784., (double) 11./84.};
	static const double b[7]  = {(double)5179./57600., 0.,(double)7571./16695.,(double)393./640.,
					      (double)-92097./339200., (double) 187./2100., (double) 1./40.};

	//dxdt(in, _dxdt0, H);

	for(int i=0;i<l;++i){
		out[i] = in[i];
		out[i] += _dxdt0[i]*a1[0]*dt;
	}
	dxdt(out, _dxdt1, H);

	for(int i=0;i<l;++i){
		out[i] = in[i];
		out[i] += _dxdt0[i]*a2[0]*dt;
		out[i] += _dxdt1[i]*a2[1]*dt;
	}
	dxdt(out, _dxdt2, H);

	for(int i=0;i<l;++i){
		out[i] = in[i];
		out[i] += _dxdt0[i]*a3[0]*dt;
		out[i] += _dxdt1[i]*a3[1]*dt;
		out[i] += _dxdt2[i]*a3[2]*dt;
	}
	dxdt(out, _dxdt3, H);
	
	for(int i=0;i<l;++i){
		out[i] = in[i];
		out[i] += _dxdt0[i]*a4[0]*dt;
		out[i] += _dxdt1[i]*a4[1]*dt;
		out[i] += _dxdt2[i]*a4[2]*dt;
		out[i] += _dxdt3[i]*a4[3]*dt;
	}
	dxdt(out, _dxdt4, H);

	for(int i=0;i<l;++i){
		out[i] = in[i];
		out[i] += _dxdt0[i]*a5[0]*dt;
		out[i] += _dxdt1[i]*a5[1]*dt;
		out[i] += _dxdt2[i]*a5[2]*dt;
		out[i] += _dxdt3[i]*a5[3]*dt;
		out[i] += _dxdt4[i]*a5[4]*dt;
	}
	dxdt(out, _dxdt5, H);
	
	for(int i=0;i<l;++i){
		out[i] = in[i];
		out[i] += _dxdt0[i]*a6[0]*dt;
		//out[i] += _dxdt1[i]*a6[1]*dt;
		out[i] += _dxdt2[i]*a6[2]*dt;
		out[i] += _dxdt3[i]*a6[3]*dt;
		out[i] += _dxdt4[i]*a6[4]*dt;
		out[i] += _dxdt5[i]*a6[5]*dt;
	}
	dxdt(out, _dxdt6, H);
	
	for(int i=0;i<l;++i){
		_errout[i] = in[i];
		_errout[i] += _dxdt0[i]*b[0]*dt;
		//_errout[i] += _dxdt1[i]*b[1]*dt;
		_errout[i] += _dxdt2[i]*b[2]*dt;
		_errout[i] += _dxdt3[i]*b[3]*dt;
		_errout[i] += _dxdt4[i]*b[4]*dt;
		_errout[i] += _dxdt5[i]*b[5]*dt;
		_errout[i] += _dxdt6[i]*b[6]*dt;
	}

	double temp;
	err[0] = 0;
	err[1] = 0;
	err[2] = 0;
	for(int i=0;i<l;++i){
		temp = std::abs(out[i].real() - _errout[i].real());
		temp = temp /(atol+rtol*std::abs(out[i].real()));
		if(err[0]<temp) err[0] = temp;

		temp = std::abs(out[i].imag() - _errout[i].imag());
		temp = temp /(atol+rtol*std::abs(out[i].imag()));
		if(err[1]<temp) err[1] = temp;
		
		temp = std::abs(out[i] - _errout[i]);
		temp = temp /(atol+rtol*std::abs(out[i]));
		if(err[2]<temp) err[2] = temp;
	}

	//delete[] _dxdt0;
	delete[] _dxdt1;
	delete[] _dxdt2;
	delete[] _dxdt3;
	delete[] _dxdt4;
	delete[] _dxdt5;
	//delete[] _dxdt6;
	delete[] _errout;
}

class test{
public:
    test(){};
    double run(double t, void (*f)(double, double*)){
        double out[2] = {0., 0.};
        f(t, out);
        return out[0]*1000+out[1];
    };
};