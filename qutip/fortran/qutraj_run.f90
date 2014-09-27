!
! TODO:
!

module qutraj_run
  !
  ! This is the main module on the fortran side of things
  !

  use qutraj_precision
  use qutraj_general
  use qutraj_hilbert
  use qutraj_evolve
  use qutraj_linalg
  use mt19937
  use linked_list

  implicit none

  !
  ! Data defining the problem
  !
  ! Invisible to python: hamilt, c_ops, e_opts, ode
  ! (because f2py can't handle derived types)
  !

  !type(operat) :: hamilt
  !type(operat), allocatable :: c_ops(:), e_ops(:)
  !type(options) :: ode

  real(wp), allocatable :: tlist(:)
  complex(wp), allocatable :: psi0(:)

  integer :: ntraj=1
  !integer :: norm_steps = 5
  !real(wp) :: norm_tol = 0.001
  integer :: n_c_ops = 0
  integer :: n_e_ops = 0
  logical :: average_states = .true.
  logical :: average_expect = .true.

  ! Optional ode options, 0 means use default values
  integer :: order=0,nsteps=0
  double precision :: first_step=0,min_step=0,max_step=0

  ! Solution
  ! format:
  ! all states: sol(1,trajectory,time,y(:))
  ! all expect: sol(e_ops(i),trajectory,time,expecation value)
  ! avg. expect: sol(e_ops(i),1,time,expectation value)
  ! if returning averaged dense density matrices:
  ! sol(1,time,rho_i,rho_j)
  complex(wp), allocatable :: sol(:,:,:,:)
  ! if returning averaged density matrices in sparse CSR format,
  ! use the following solution array and get_rho_sparse instead.
  type(operat), allocatable :: sol_rho(:)

  ! use sparse density matrices during computation?
  logical :: rho_return_sparse = .true.

  ! temporary storage for csr matrix, available for python
  ! this is needed because you can't send assumed
  ! shape arrays to python
  complex(wp), allocatable :: csr_val(:)
  integer, allocatable :: csr_col(:), csr_ptr(:)
  integer :: csr_nrows,csr_ncols

  ! Collapse times and integer denoting which operator did it
  ! temporary storage available for python
  real(wp), allocatable :: col_times(:)
  integer, allocatable :: col_which(:)
  ! data stored internally in linked lists, one per trajectory
  type(linkedlist_real), allocatable :: ll_col_times(:)
  type(linkedlist_int), allocatable :: ll_col_which(:)

  ! Integer denoting the type of unravelling
  ! 1 for no collapse operatros
  ! 2 for jump unravelling
  ! diffusive unravellings to be implemented
  integer :: unravel_type = 2

  ! Stuff needed for partial trace
  integer, allocatable :: psi0_dims1(:),ptrace_sel(:)
  integer :: rho_reduced_dim=0
  ! Calculate average entropy of reduced state over trajectories?
  logical :: calc_entropy = .false.
  real(wp), allocatable :: reduced_state_entropy(:)

  !
  ! Interfaces
  !

  interface finalize
    module procedure options_finalize
  end interface

  contains

  !
  ! Initialize problem
  !

  subroutine init_tlist(val,n)
    use qutraj_precision
    real(wp), intent(in) :: val(n)
    integer, intent(in) :: n
    call new(tlist,val)
  end subroutine

  subroutine init_psi0(val,n)
    use qutraj_precision
    complex(wp), intent(in) :: val(n)
    integer, intent(in) :: n
    call new(psi0,val)
  end subroutine

  subroutine init_ptrace_stuff(dims,sel,reduced_dim,ndims,nsel)
    integer, intent(in) :: dims(ndims),sel(nsel),reduced_dim
    integer, intent(in) :: ndims, nsel
    !complex(wp), allocatable :: rho(:,:)
    !real(wp) :: S
    call new(psi0_dims1,dims)
    call new(ptrace_sel,sel)
    rho_reduced_dim = reduced_dim
    !allocate(rho(rho_reduced_dim,rho_reduced_dim))
    !call ptrace_pure(psi0,rho,ptrace_sel,psi0_dims1)
    !write(*,*) rho
    !call entropy(rho,S)
    !write(*,*) S
  end subroutine

  subroutine init_hamiltonian(val,col,ptr,m,k,nnz,nptr)
    ! Hamiltonian is assumed to be given as
    ! -i*(H - i/2 sum c_ops(i)^* c_ops(i))
    use qutraj_precision
    integer, intent(in) :: nnz,nptr,m,k
    complex(wp), intent(in)  :: val(nnz)
    integer, intent(in) :: col(nnz),ptr(nptr)
    call new(hamilt,val,col,ptr,m,k)
  end subroutine

  subroutine init_c_ops(i,n,val,col,ptr,m,k,first,nnz,nptr)
    use qutraj_precision
    integer, intent(in) :: i,n
    integer, intent(in) :: nnz,nptr,m,k
    complex(wp), intent(in) :: val(nnz)
    integer, intent(in) :: col(nnz),ptr(nptr)
    logical, optional :: first
    if (.not.present(first)) then
      first = .false.
    endif
    if (first) then
      call new(c_ops,n)
    endif
    if (.not.allocated(c_ops)) then
      call error('init_c_ops: c_ops not allocated. call with first=True first.')
    endif
    n_c_ops = n
    call new(c_ops(i),val,col,ptr,m,k)
  end subroutine

  subroutine init_e_ops(i,n,val,col,ptr,m,k,first,nnz,nptr)
    use qutraj_precision
    integer, intent(in) :: i,n
    integer, intent(in) :: nnz,nptr,m,k
    complex(wp), intent(in) :: val(nnz)
    integer, intent(in) :: col(nnz),ptr(nptr)
    logical, optional :: first
    if (.not.present(first)) then
      first = .false.
    endif
    if (first) then
      call new(e_ops,n)
    endif
    if (.not.allocated(e_ops)) then
      call error('init_e_ops: e_ops not allocated. call with first=True first.')
    endif
    n_e_ops = n
    call new(e_ops(i),val,col,ptr,m,k)
  end subroutine

  subroutine init_result(neq,atol,rtol,mf,norm_steps,norm_tol,&
      lzw,lrw,liw,ml,mu,natol,nrtol)
    use qutraj_precision
    integer, intent(in) :: neq
    integer, intent(in), optional :: lzw,lrw,liw,mf,norm_steps
    integer, intent(in) :: natol,nrtol
    double precision, optional :: atol(1),rtol(1)
    real(wp), optional :: norm_tol
    integer, intent(in), optional :: ml,mu
    !integer :: istat

    ode%neq = neq
    if (lzw.ne.0) then
      ode%lzw = lzw
    endif
    if (lrw.ne.0) then
      ode%lrw = lrw
    endif
    if (liw.ne.0) then
      ode%liw = liw
    endif
    if (lrw.eq.0) then
      ode%lrw = 20+neq
    endif

    if (mf==0 .or. mf==10) then
      ! assuming non-stiff by default
      ode%mf=10
      if (lzw.eq.0) then
        ode%lzw = 15*neq
      endif
      if (liw.eq.0) then
        ode%liw = 30
      endif
    elseif (mf==21.or.mf==22) then
      ode%mf = mf
      if (lzw.eq.0) then
        ode%lzw = 8*neq+2*neq**2
      endif
      if (liw.eq.0) then
        ode%liw = 30+neq
      endif
    elseif (mf==24.or.mf==25) then
      ode%mf = mf
      if (lzw.eq.0) then
        ! mf=24,25 requires ml and mu
        ode%lzw = 10*neq + (3*ml + 2*mu)*neq
      endif
      if (liw.eq.0) then
        ode%liw = 30+neq
      endif
    endif

    call new(ode%zwork,ode%lzw)
    call new(ode%rwork,ode%lrw)
    call new(ode%iwork,ode%liw)
    call new(ode%atol,atol)
    call new(ode%rtol,rtol)
    if (size(ode%atol)==1) then
      ode%itol=1
    else
      ode%itol=2
    endif
    ode%iopt = 0

    if (norm_steps.ne.0) ode%norm_steps = norm_steps
    if (norm_tol.ne.0.) ode%norm_tol = norm_tol
  end subroutine

  subroutine get_rho_sparse(i)
    integer, intent(in) :: i
    call new(csr_val,sol_rho(i)%a)
    call new(csr_col,sol_rho(i)%ia1)
    call new(csr_ptr,sol_rho(i)%pb)
    csr_nrows = sol_rho(i)%m
    csr_ncols = sol_rho(i)%k
  end subroutine

  subroutine get_collapses(traj)
    integer, intent(in) :: traj
    integer :: i
    ! Turn linked lists into arrays
    call ll_to_array(ll_col_times(traj),col_times)
    call ll_to_array(ll_col_which(traj),col_which)
    if (traj==ntraj) then
      do i=1,ntraj
        call finalize(ll_col_times(i))
        call finalize(ll_col_which(i))
      enddo
      deallocate(ll_col_times,ll_col_which)
    endif
  end subroutine

  !
  ! Evolution
  !

  subroutine evolve(instanceno, rngseed, show_progress)
    ! What process # am I?
    integer, intent(in) :: instanceno, rngseed, show_progress
    double precision :: t, tout
    double complex, allocatable :: y(:),y_tmp(:),rho(:,:)
    logical :: states
    type(operat) :: rho_sparse
    integer :: istat=0,istat2=0,traj,progress
    integer :: i,j,l
    !integer :: m,n
    real(wp) :: mu,nu,S
    real(wp), allocatable :: p(:)
    ! ITASK  = An index specifying the task to be performed.
    !          Input only.  ITASK has the following values and meanings.
    !          1  means normal computation of output values of y(t) at
    !             t = TOUT (by overshooting and interpolating).
    !          2  means take one step only and return.
    !          3  means stop at the first internal mesh point at or
    !             beyond t = TOUT and return.
    !          4  means normal computation of output values of y(t) at
    !             t = TOUT but without overshooting t = TCRIT.
    !             TCRIT must be input as RWORK(1).  TCRIT may be equal to
    !             or beyond TOUT, but not behind it in the direction of
    !             integration.  This option is useful if the problem
    !             has a singularity at or beyond t = TCRIT.
    !          5  means take one step, without passing TCRIT, and return.
    !             TCRIT must be input as RWORK(1).
    !
    !          Note:  If ITASK = 4 or 5 and the solver reaches TCRIT
    !          (within roundoff), it will return T = TCRIT (exactly) to
    !          indicate this (unless ITASK = 4 and TOUT comes before 
    !          TCRIT, in which case answers at T = TOUT are returned 
    !          first).

    ! States or expectation values
    !if (n_e_ops == 0 .and. .not.calc_entropy) then
    if (n_e_ops == 0) then
      states = .true.
    else
      states = .false.
    endif
    ! Allocate solution array
    if (allocated(sol)) then
      deallocate(sol,stat=istat)
      if (istat.ne.0) then
        call error("evolve: could not deallocate.",istat)
      endif
    endif

    if (states) then
      if (average_states) then
        if (rho_return_sparse) then
          call new(sol_rho,size(tlist))
          call new(rho_sparse,1,1)
        else
          if (rho_reduced_dim == 0) then
            ! Not doing partial trace
            allocate(sol(1,size(tlist),ode%neq,ode%neq),stat=istat)
            allocate(rho(ode%neq,ode%neq), stat=istat2)
          else
            ! Doing partial trace
            allocate(sol(1,size(tlist),&
              rho_reduced_dim,rho_reduced_dim),stat=istat)
            allocate(rho(rho_reduced_dim,rho_reduced_dim), stat=istat2)
          endif
          sol = (0.,0.)
          rho = (0.,0.)
        endif
      else
        allocate(sol(1,ntraj,size(tlist),ode%neq), stat=istat)
        sol = (0.,0.)
      endif
    
    elseif (n_e_ops>0) then
      if (average_expect) then
        allocate(sol(n_e_ops,1,size(tlist),1), stat=istat)
        sol = (0.,0.)
      else
        allocate(sol(n_e_ops,ntraj,size(tlist),1), stat=istat)
        sol = (0.,0.)
      endif
    endif

    if (istat.ne.0) then
      call fatal_error("evolve: could not allocate sol.", istat)
    endif
 
    if (istat2.ne.0) then
      call fatal_error("evolve: could not allocate rho.", istat2)
    endif

    ! Array for average entropy
    if (calc_entropy) then
      if (.not.allocated(rho)) then
        allocate(rho(rho_reduced_dim, rho_reduced_dim), stat=istat2)
      endif
      call new(reduced_state_entropy,size(tlist))
      reduced_state_entropy = 0.
    endif

    ! Allocate linked lists for collapse times and operators
    if (allocated(ll_col_times)) then
      deallocate(ll_col_times, stat=istat)
      if (istat.ne.0) then
        call error("evolve: could not deallocate ll_col_times.", istat)
      endif
    endif

    allocate(ll_col_times(ntraj), stat=istat)
    if (istat.ne.0) then
      call fatal_error("evolve: could not allocate ll_col_times.", istat)
    endif

    if (allocated(ll_col_which)) then
      deallocate(ll_col_which, stat=istat)
      if (istat.ne.0) then
        call fatal_error("evolve: could not deallocate ll_col_which.", istat)
      endif
    endif

    allocate(ll_col_which(ntraj), stat=istat)
    if (istat.ne.0) then
      call fatal_error("evolve: could not allocate ll_col_which.", istat)
    endif

    ! Allocate work arrays
    call new(y,ode%neq)
    call new(y_tmp,ode%neq)
    ! Allocate tmp array for jump probabilities
    call new(p,n_c_ops)
    ! Initalize rng
    call init_genrand(rngseed)

    ! Initial ode setup
    if (unravel_type==1) then
      ! integrate one until specified time, w/o overshooting
      ode%itask = 4
    elseif (unravel_type==2) then
      ! integrate one step at the time, w/o overshooting
      ode%itask = 5
    endif
    ! set optinal arguments
    ! see zvode.f
    ode%rwork = 0.0
    ode%iwork = 0
    ode%rwork(5) = first_step
    ode%rwork(6) = max_step
    ode%rwork(7) = min_step
    ode%iwork(5) = order
    ode%iwork(6) = nsteps
    ode%iopt = 1
    ! first call to zvode
    ode%istate = 1

    ! Loop over trajectories
    progress = 1
    do traj=1,ntraj
      ! two random numbers
      mu = grnd()
      nu = grnd()
      ! First call to zvode
      ode%istate = 1
      ! Initial values
      y = psi0
      ! Initial value of indep. variable
      t = tlist(1)
      do i=1,size(tlist)
        ! Solution wanted at
        if (i==1) then
          ! found this to be necessary due to round off error
          tout = t
        else
          tout = tlist(i)
        endif
        select case(unravel_type)
        case(1)
          call evolve_nocollapse(t,tout,y,y_tmp,ode)
        case(2)
          call evolve_jump(t,tout,y,y_tmp,p,mu,nu,&
            ll_col_times(traj),ll_col_which(traj),ode)
        case default
          call fatal_error('Unknown unravel type.')
        end select
        y_tmp = y
        call normalize(y_tmp)

        ! Compute solution
        if (rho_reduced_dim.ne.0) then
           call ptrace_pure(y_tmp,rho,ptrace_sel,psi0_dims1)
        endif
        if (states) then
          if (average_states) then
            ! construct density matrix
            if (rho_return_sparse) then
              call densitymatrix_sparse(y_tmp,rho_sparse)
              if (traj==1) then
                sol_rho(i) = rho_sparse
              else
                sol_rho(i) = sol_rho(i) + rho_sparse
              endif
            else
              if (rho_reduced_dim == 0) then
                call densitymatrix_dense(y_tmp,rho)
              !else
              !  call ptrace_pure(y_tmp,rho,ptrace_sel,psi0_dims1)
              endif
              sol(1,i,:,:) = sol(1,i,:,:) + rho
            endif
          else
            sol(1,traj,i,:) = y_tmp
          endif
        
        else
          if (average_expect) then
            do l=1,n_e_ops
              sol(l,1,i,1) = sol(l,1,i,1)+braket(y_tmp,e_ops(l)*y_tmp)
            enddo
          else
            do l=1,n_e_ops
              sol(l,traj,i,1) = braket(y_tmp,e_ops(l)*y_tmp)
            enddo
          endif
        endif
        if (calc_entropy) then
          call entropy(rho,S)
          reduced_state_entropy(i) = reduced_state_entropy(i) + S
        endif
        ! End time loop
      enddo
      ! Indicate progress
      if (show_progress == 1 .and. instanceno == 1 .and. traj.ge.progress*ntraj/10.0) then
        write(*,*) "progress of process 1: ", progress*10, "%"
        progress=progress+1
      endif
      ! End loop over trajectories
    enddo
    ! Normalize
    if (average_states) then
      if (states .and. rho_return_sparse) then
        do j=1,size(sol_rho)
          sol_rho(j) = (1._wp/ntraj)*sol_rho(j)
        enddo
      endif
    endif
    if (allocated(sol) .and. average_expect) then
        sol = (1._wp/ntraj)*sol
    endif
    if (calc_entropy .and. (average_states .or. average_expect)) then
        reduced_state_entropy = (1._wp/ntraj)*reduced_state_entropy
    endif
    
    ! Deallocate
    call finalize(y)
    call finalize(y_tmp)
    call finalize(p)
    if (allocated(rho)) then
      deallocate(rho)
    endif
  end subroutine

  !
  ! Misc
  !

  ! Deallocate stuff

  subroutine options_finalize(this)
    type(options), intent(inout) :: this
    integer :: istat
    if (allocated(this%zwork)) then
      deallocate(this%zwork,stat=istat)
      if (istat.ne.0) then
        call error("options_finalize: could not deallocate.",istat)
      endif
    endif
    if (allocated(this%rwork)) then
      deallocate(this%rwork,stat=istat)
      if (istat.ne.0) then
        call error("options_finalize: could not deallocate.",istat)
      endif
    endif
    if (allocated(this%iwork)) then
      deallocate(this%iwork,stat=istat)
      if (istat.ne.0) then
        call error("options_finalize: could not deallocate.",istat)
      endif
    endif
    if (allocated(this%atol)) then
      deallocate(this%atol,stat=istat)
      if (istat.ne.0) then
        call error("options_finalize: could not deallocate.",istat)
      endif
    endif
    if (allocated(this%rtol)) then
      deallocate(this%rtol,stat=istat)
      if (istat.ne.0) then
        call error("options_finalize: could not deallocate.",istat)
      endif
    endif
  end subroutine

  subroutine finalize_work
    !integer :: istat=0
    call finalize(psi0)
    call finalize(hamilt)
    call finalize(c_ops)
    call finalize(e_ops)
    call finalize(ode)
  end subroutine

  subroutine finalize_sol
    integer :: istat=0
    call finalize(tlist)
    call finalize(sol_rho)
    if (allocated(ll_col_times)) then
      deallocate(ll_col_times,stat=istat)
    endif
    if (istat.ne.0) then
      call error("finalize_sol: could not deallocate.",istat)
    endif
    if (allocated(ll_col_which)) then
      deallocate(ll_col_which,stat=istat)
    endif
    if (istat.ne.0) then
      call error("finalize_sol: could not deallocate.",istat)
    endif
    if (allocated(sol)) then
      deallocate(sol,stat=istat)
    endif
    if (istat.ne.0) then
      call error("finalize_sol: could not deallocate.",istat)
    endif
  end subroutine

  ! Misc

  subroutine test_real_precision
    use qutraj_precision
    real(wp) :: b,a
    integer :: i
    write(*,*) "wp=",wp
    b = 1.0
    a = 1.0
    i = 1
    do while (b.ne.b+a)
      a = a*0.1
      if (b==b+a) then
        write(*,*) "number of decimals working precision: ",i-1
      endif
      i = i+1
    enddo
  end subroutine

end module
