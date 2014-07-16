module qutraj_evolve
  use qutraj_precision
  use qutraj_general
  use qutraj_hilbert
  use linked_list
  use mt19937

  implicit none

  !
  ! Types
  !

  type options
    ! No. of ODES
    integer :: neq=1
    ! work array zwork should have length 15*neq for non-stiff
    integer :: lzw = 0
    double complex, allocatable :: zwork(:)
    ! work array rwork should have length 20+neq for non-siff
    integer :: lrw = 0
    double precision, allocatable :: rwork(:)
    ! work array iwork should have length 30 for non-stiff
    integer :: liw = 0
    integer, allocatable :: iwork(:)
    ! method flag mf should be 10 for non-stiff
    integer :: mf = 10
    ! arbitrary real/complex and int array for user def input to rhs
    double complex :: rpar(1)
    integer :: ipar(1)
    ! abs. tolerance, rel. tolerance 
    double precision, allocatable :: atol(:), rtol(:)
    ! iopt=number of optional inputs, itol=1 for atol scalar, 2 otherwise
    integer :: iopt, itol
    ! task and state of solver
    integer :: itask, istate
    ! tolerance for trying to find correct jump times
    integer :: norm_steps = 5
    real(wp) :: norm_tol = 0.001
  end type

  !
  ! Public data
  !

  type(operat) :: hamilt
  type(operat), allocatable :: c_ops(:), e_ops(:)
  type(options) :: ode

  ! Hermitian conjugated operators
  type(operat), allocatable :: c_ops_hc(:)

  contains

  !
  ! Evolution subs
  !

  subroutine evolve_nocollapse(t,tout,y,y_tmp,ode)
    double complex, intent(inout) :: y(:),y_tmp(:)
    double precision, intent(inout) :: t, tout
    type(options) :: ode

    ! integrate up to tout without overshooting
    ode%rwork(1) = tout

    call nojump(y,t,tout,ode%itask,ode)
    if (ode%istate.lt.0) then
      write(*,*) "zvode error: istate=",ode%istate
      !stop
    endif
  end subroutine

  subroutine evolve_jump(t,tout,y,y_tmp,p,mu,nu,&
    ll_col_times,ll_col_which,ode)
    !
    ! Evolve quantum trajectory y(t) to y(tout) using ``jump'' method
    !
    ! Input: t, tout, y
    ! Work arrays: y_tmp, p
    ! mu, nu: two random numbers
    !
    double complex, intent(inout) :: y(:),y_tmp(:)
    double precision, intent(inout) :: t, tout
    real(wp), intent(inout) :: p(:)
    real(wp), intent(inout) :: mu,nu
    type(linkedlist_real), intent(inout) :: ll_col_times
    type(linkedlist_int), intent(inout) :: ll_col_which
    type(options) :: ode
    double precision :: t_prev, t_final, t_guess
    integer :: j,k
    integer :: cnt
    real(wp) :: norm2_psi,norm2_prev,norm2_guess,sump
    ! logical, save :: first = .true.

    ode%rwork(1) = tout
    norm2_psi = abs(braket(y,y))
    do while(t<tout)
      t_prev = t
      y_tmp = y
      norm2_prev = norm2_psi
      call nojump(y,t,tout,ode%itask,ode)
      if (ode%istate.lt.0) then
        write(*,*) "zvode error: istate=",ode%istate
        !stop
      endif
      ! prob of nojump
      norm2_psi = abs(braket(y,y))
      if (norm2_psi.le.mu) then
        ! jump happened
        ! find collapse time to specified tolerance
        t_final = t
        cnt=1
        do k=1,ode%norm_steps
          !t_guess=t_prev+(mu-norm2_prev)&
          !  /(norm2_psi-norm2_prev)*(t_final-t_prev)
          t_guess=t_prev+log(norm2_prev/mu)&
            /log(norm2_prev/norm2_psi)*(t_final-t_prev)
          if (t_guess<t_prev .or. t_guess>t_final) then
            t_guess = t_prev+0.5*(t_final-t_prev)
          endif
          y = y_tmp
          t = t_prev
          call nojump(y,t,t_guess,1,ode)
          if (ode%istate.lt.0) then
            write(*,*) "zvode failed after adjusting step size. istate=",ode%istate
            !stop
          endif
          norm2_guess = abs(braket(y,y))
          if (abs(mu-norm2_guess) < ode%norm_tol*mu) then
              exit
          elseif (norm2_guess < mu) then
              ! t_guess is still > t_jump
              t_final=t_guess
              norm2_psi=norm2_guess
          else
              ! t_guess < t_jump
              t_prev=t_guess
              y_tmp=y
              norm2_prev=norm2_guess
          endif
          cnt = cnt+1
        enddo
        if (cnt > ode%norm_steps) then
          call error("Norm tolerance not reached. Increase accuracy of ODE solver or norm_steps.")
        endif
        ! determine which jump
        do j=1,size(c_ops)
          y_tmp = c_ops(j)*y
          p(j) = abs(braket(y_tmp,y_tmp))
        enddo
        p = p/sum(p)
        sump = 0
        do j=1,size(c_ops)
          if ((sump <= nu) .and. (nu < sump+p(j))) then
            y = c_ops(j)*y
            ! Append collapse time and operator # to linked lists
            call append(ll_col_times,t)
            call append(ll_col_which,j)
          endif
          sump = sump+p(j)
        enddo
        ! new random numbers
        mu = grnd()
        nu = grnd()
        ! normalize y
        call normalize(y)
        ! reset, first call to zvode
        ode%istate = 1
      endif
    enddo
  end subroutine

  subroutine nojump(y,t,tout,itask,ode)
    ! evolve with effective hamiltonian
    type(options), intent(in) :: ode
    double complex, intent(inout) :: y(:)
    double precision, intent(inout) :: t
    double precision, intent(in) :: tout
    integer, intent(in) :: itask
    !integer :: istat

    call zvode(rhs,ode%neq,y,t,tout,ode%itol,ode%rtol,ode%atol,&
      itask,ode%istate,ode%iopt,ode%zwork,ode%lzw,ode%rwork,ode%lrw,&
      ode%iwork,ode%liw,dummy_jac,ode%mf,ode%rpar,ode%ipar)
  end subroutine

  !
  ! RHS for zvode
  !

  subroutine rhs (neq, t, y, ydot, rpar, ipar)
    ! evolve with effective hamiltonian
    complex(wp) :: y(neq), ydot(neq),rpar
    real(wp) :: t
    integer :: ipar,neq
    ydot = (hamilt*y)
  end subroutine

  subroutine dummy_jac (neq, t, y, ml, mu, pd, nrpd, rpar, ipar)
    ! dummy jacobian for zvode
    complex(wp) :: y(neq), pd(nrpd,neq), rpar
    real(wp) :: t
    integer :: neq,ml,mu,nrpd,ipar
    return
  end subroutine

  !
  ! Diffusive unravelling evolution
  !

  !subroutine evolve_platen(psi,delta_t)
  !  ! TODO: Clean up use of temporary state vectors
  !  ! Diffusive solution, Platen scheme
  !  ! Evolve for a small time step delta_t
  !  ! State, inout, normalized
  !  complex(wp), intent(inout) :: psi(:)
  !  real(wp), intent(in) :: delta_t
  !  real(wp) :: p1,p2,pnj,dw
  !  integer :: i,j
  !  complex(wp), allocatable :: psi_n,dpsi1,dpsi2
  !  complex(wp), allocatable :: psi_tilde,psi_plus,psi_min

  !  call new(psi_n,size(psi))
  !  call new(dpsi1,size(psi))
  !  call new(dpsi2,size(psi))
  !  call new(psi_tilde,size(psi))
  !  call new(psi_plus,size(psi))
  !  call new(psi_min,size(psi))

  !  psi_n = psi

  !  ! Hamiltonian term
  !  !call hamiltonian(hamiltonian_id,psi,dpsi1)
  !  dpsi1 = -ii*(hamilt*psi)
  !  psi_tilde = psi + delta_t*dpsi1
  !  psi_n = psi_n + 0.5*delta_t*dpsi1
  !  !call hamiltonian(hamiltonian_id,psi_tilde,dpsi1)
  !  dpsi1 = (-ii)*(hamilt*psi_tilde)
  !  psi_n = psi_n + 0.5*delta_t*dpsi1

  !  do j=1,n_c_ops
  !    call schrod_d1_bp(j,psi,dpsi1)
  !    call schrod_d2_bp(j,psi,dpsi2)

  !    dw = sqrt(delta_t)*gaussran(rngseed,rngseed)
  !    psi_tilde = psi + delta_t*dpsi1 + dw*dpsi2
  !    psi_plus = psi + delta_t*dpsi1 + sqrt(delta_t)*dpsi2
  !    psi_min = psi + delta_t*dpsi1 - sqrt(delta_t)*dpsi2

  !    psi_n = psi_n + delta_t/2.0*dpsi1
  !    psi_n = psi_n + dw/2.0*dpsi2

  !    call schrod_d1_bp(j,psi_tilde,dpsi1)
  !    psi_n = psi_n + 0.5*delta_t*dpsi1
  !    call schrod_d2_bp(j,psi_plus,dpsi2)
  !    psi_n = psi_n + (dw + (dw*dw-delta_t)/sqrt(delta_t))/4.0*dpsi2
  !    call schrod_d2_bp(j,psi_min,dpsi2)
  !    psi_n = psi_n + (dw - (dw*dw-delta_t)/sqrt(delta_t))/4.0*dpsi2
  !  enddo
  !  call normalize(psi_n)
  !  psi = psi_n
  !end subroutine

  !subroutine d1_bp(i,psi,dpsi)
  !  ! D1 term from Breuer & Pettruccione
  !  ! Return D1 |Psi(t)> in dpsi, for jump-operator c_ops(i)
  !  ! B&P p. 331 eq (6.181)
  !  complex(wp), intent(in) :: psi(:)
  !  complex(wp), intent(out) :: dpsi(:)
  !  !complex(wp), allocatable :: psi_tmp(:)
  !  integer, intent(in) :: i
  !  complex(wp) :: tmp1, tmp2
  !  !call new(psi_tmp,size(psi))
  !  tmp1 = braket(psi,c_ops(i)*psi)
  !  tmp2 = braket(psi,c_ops(i)*psi)
  !  dpsi = 0.5_wp*(tmp1+tmp2)*(c_ops(i)*psi)
  !  dpsi = dpsi-(0.5_wp)*(c_ops_hc(i)*(c_ops(i)*psi))
  !  dpsi = dpsi-(0.125_wp*(tmp1+tmp2)*(tmp1+tmp2))*psi
  !end subroutine

  !subroutine d2_bp(i,psi,dpsi)
  !  ! D2 term from Breuer & Pettruccione
  !  ! Return D2 |Psi(t)> in dpsi, for jump-operator c_ops(i)
  !  ! B&P p. 331 eq (6.181)
  !  complex(wp), intent(in) :: psi(:)
  !  complex(wp), intent(out) :: dpsi(:)
  !  integer, intent(in) :: i
  !  complex(wp) :: tmp1, tmp2
  !  tmp1 = braket(psi,c_ops(i)*psi)
  !  tmp2 = braket(psi,c_ops_hc(i)*psi)
  !  dpsi = c_ops(i)*psi
  !  dpsi = dpsi - (0.5_wp*(tmp1+tmp2))*psi
  !end subroutine

end module

