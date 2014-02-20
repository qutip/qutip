module qutraj_hilbert
  !
  ! Module implementing and providing abstraction of
  ! the Hilbert space
  !
  ! States are complex 1d arrays, and
  ! operators are CSR matrices
  !

  use qutraj_precision
  use qutraj_general

  implicit none

  !
  ! Types
  !

  type operat
    ! Operators are represented as spare matrices
    ! stored in compressed row format (CSR)

    ! m = number of rows, k = number of cols
    ! (should have m=k for operators!)
    integer :: m,k
    ! number of values
    integer :: nnz
    !! compression format is CSR
    !character*5 :: fida = 'CSR'
    !! base: Fortran or C base
    !integer :: base = 1 !
    !! diag: 'U' for un-stored diag entries, assumed to be one
    !character*11 :: diag = 'N'
    !! typem: 'S' for symmetric, 'H' for Hermitian
    !character*11 :: typem = 'G'
    !!both/lower/upper half of matrix specified
    !character*11 :: part = 'B'
    ! values
    complex(wp), allocatable :: a(:)
    integer, allocatable :: ia1(:),pb(:)!,pe(:)
    ! notice: pe(i) = pb(i+1)-1
  end type

  !
  ! Interfaces
  !

  interface new
    module procedure state_init
    module procedure state_init2_wp
    module procedure operat_init
    module procedure operat_init2_wp
    module procedure operat_list_init
  end interface

  interface finalize
    module procedure state_finalize
    module procedure operat_finalize
    module procedure operat_list_finalize
  end interface

  interface assignment(=)
    module procedure operat_operat_eq
  end interface

  interface operator(*)
    module procedure operat_state_mult
    module procedure real_operat_mult
    module procedure operat_operat_mult
  end interface

  interface operator(+)
    module procedure operat_operat_add
  end interface

  !
  ! Subs and funcs
  !

  contains

  !
  ! Initializers & finalizers
  !

  subroutine state_init(this,n)
    complex(wp), allocatable :: this(:)
    integer, intent(in) :: n
    integer :: istat=0
    if (allocated(this)) then
      deallocate(this,stat=istat)
    endif
    allocate(this(n),stat=istat)
    if (istat.ne.0) then
      call fatal_error("state_init: could not allocate.",istat)
    endif
  end subroutine

  subroutine state_init2_wp(this,val)
    complex(wp), allocatable :: this(:)
    complex(wp), intent(in) :: val(:)
    call state_init(this,size(val))
    this = val
  end subroutine

  subroutine state_finalize(this)
    complex(wp), allocatable :: this(:)
    integer :: istat=0
    if (allocated(this)) then
      deallocate(this,stat=istat)
    endif
    if (istat.ne.0) then
      call error("state_finalize: could not deallocate.",istat)
    endif
  end subroutine

  subroutine operat_init(this,nnz,nptr)
    ! todo: add special support for Hermitian matrix
    type(operat), intent(out) :: this
    integer, intent(in) :: nnz,nptr
    integer :: istat=0,nnz_,nptr_
    if (allocated(this%a)) then
      deallocate(this%a,stat=istat)
    endif
    if (allocated(this%ia1)) then
      deallocate(this%ia1,stat=istat)
    endif
    if (allocated(this%pb)) then
      deallocate(this%pb,stat=istat)
    endif
    nnz_ = nnz
    nptr_ = nptr
    if (nnz==0) nnz_= 1
    if (nptr==0) nptr_= 1
    this%nnz = nnz_
    allocate(this%a(nnz_),stat=istat)
    if (istat.ne.0) then
      call fatal_error("operat_init: could not allocate.",istat)
    endif
    allocate(this%ia1(nnz_),stat=istat)
    if (istat.ne.0) then
      call fatal_error("operat_init: could not allocate.",istat)
    endif
    allocate(this%pb(nptr_),stat=istat)
    if (istat.ne.0) then
      call fatal_error("operat_init: could not allocate.",istat)
    endif
    ! Set to zero
    this%a = (0.,0.)
    !this%ia1 = 1
    !this%pb = 1
    !this%m = 1
    !this%k = 1
    ! Set default parameters
    !this%fida = 'CSR'
    !this%base = 1 ! fortran base
    !this%diag = 'N'
    !this%typem = 'G'
    !this%part = 'B'
  end subroutine

  subroutine operat_init2_wp(this,val,col,ptr,m,k)
    integer, intent(in) :: m,k
    type(operat), intent(out) :: this
    complex(wp), intent(in) :: val(:)
    integer, intent(in) :: col(:),ptr(:)
    ! integer :: i
    if (size(val)==0) then
      call operat_init(this,1,1)
      this%m = 1
      this%k = 1
      this%a = (/(0.,0.)/)
      this%ia1 = (/1/)
      this%pb = (/1,2/)
    else
      call operat_init(this,size(val),size(ptr))
      this%m = m
      this%k = k
      this%a = val
      this%ia1 = col
      this%pb = ptr
    endif
  end subroutine

  subroutine operat_list_init(this,n)
    type(operat), intent(inout), allocatable :: this(:)
    integer, intent(in) :: n
    integer :: istat
    if (allocated(this)) then
      deallocate(this,stat=istat)
    endif
    allocate(this(n),stat=istat)
    if (istat.ne.0) then
      call fatal_error("operat_list_init: could not allocate.",istat)
    endif
  end subroutine

  subroutine operat_finalize(this)
    type(operat), intent(inout) :: this
    integer :: istat=0
    if (allocated(this%a)) then
      deallocate(this%a,stat=istat)
      if (istat.ne.0) then
        call error("operat_finalize: could not deallocate.",istat)
      endif
    endif
    if (allocated(this%ia1)) then
      deallocate(this%ia1,stat=istat)
      if (istat.ne.0) then
        call error("operat_finalize: could not deallocate.",istat)
      endif
    endif
    if (allocated(this%pb)) then
      deallocate(this%pb,stat=istat)
      if (istat.ne.0) then
        call error("operat_finalize: could not deallocate.",istat)
      endif
    endif
  end subroutine

  subroutine operat_list_finalize(this)
    type(operat), intent(inout), allocatable :: this(:)
    integer :: istat=0,i
    if (allocated(this)) then
      do i=1,size(this)
        call finalize(this(i))
      enddo
      deallocate(this,stat=istat)
    endif
    if (istat.ne.0) then
      call error("operat_list_finalize: could not deallocate.",istat)
    endif
  end subroutine

  !
  ! State/operator arithmetic
  !

  subroutine operat_operat_eq(this,a)
    type(operat), intent(out) :: this
    type(operat), intent(in) :: a
    call new(this,a%a,a%ia1,a%pb,a%m,a%k)
  end subroutine

  function operat_operat_add(a,b)
    type(operat) :: operat_operat_add
    type(operat), intent(in) :: a,b
    type(operat) :: c,d
    integer :: nzmax,ierr
    integer, allocatable :: iw(:)
    call new(iw,a%k)
    nzmax = size(a%a) + size(b%a)
    call new(c,nzmax,a%m+1)
    call aplb(a%m,a%k,1,a%a,a%ia1,a%pb,b%a,b%ia1,b%pb,c%a,c%ia1,c%pb,&
      nzmax,iw,ierr)
    if (ierr.ne.0) then
      call error('operat_operat_add',ierr)
    endif
    !call finalize(iw)
    nzmax = count(c%a.ne.0)
    c%pb(size(c%pb)) = nzmax+1
    call new(d,c%a(1:nzmax),c%ia1(1:nzmax),c%pb,a%m,b%k)
    operat_operat_add = d
  end function

  function operat_state_mult(oper,psi)
    complex(wp), intent(in) :: psi(:)
    type(operat), intent(in) :: oper
    complex(wp):: operat_state_mult(size(psi))
    complex(wp), allocatable :: tmp(:)
    integer :: ierr
    call new(tmp,size(psi))
    call sparse_mv_mult(oper,psi,tmp,ierr)
    if (ierr.ne.0) then
      call error("operate_state_mult: error",ierr)
    endif
    operat_state_mult = tmp
    call finalize(tmp)
  end function

  function real_operat_mult(r,a)
    type(operat) :: real_operat_mult
    real(wp), intent(in) :: r
    type(operat), intent(in) :: a
    type(operat) :: c
    call new(c,r*a%a,a%ia1,a%pb,a%m,a%k)
    real_operat_mult = c
  end function

  function operat_operat_mult(a,b)
    type(operat) :: operat_operat_mult
    type(operat), intent(in) :: a,b
    type(operat) :: c,d
    integer :: nzmax,ierr
    integer, allocatable :: iw(:)
    call new(iw,b%k)
    nzmax = a%nnz*b%nnz
    call new(c,nzmax,a%m+1)
    call amub(a%m,b%k,1,a%a,a%ia1,a%pb,b%a,b%ia1,b%pb,&
      c%a,c%ia1,c%pb,nzmax,iw,ierr)
    if (ierr.ne.0) call error('operat_operat_mult',ierr)
    !call finalize(iw)
    nzmax = count(c%a.ne.0)
    c%pb(size(c%pb)) = nzmax+1
    call new(d,c%a(1:nzmax),c%ia1(1:nzmax),c%pb,a%m,b%k)
    operat_operat_mult = d
  end function

  subroutine operat_operat_mult_sub(a,b,d)
    type(operat), intent(out) :: d
    type(operat), intent(in) :: a,b
    type(operat) :: c
    integer :: nzmax,ierr
    integer, allocatable :: iw(:)
    call new(iw,b%k)
    nzmax = a%nnz*b%nnz
    write(*,*) 'a%m',a%m,'a%k',a%k,'a%nnz',a%nnz
    write(*,*) 'b%m',b%m,'b%k',b%k,'b%nnz',b%nnz
    call new(c,nzmax,a%m+1)
    write(*,*) 'created c'
    call amub(a%m,b%k,1,a%a,a%ia1,a%pb,b%a,b%ia1,b%pb,&
      c%a,c%ia1,c%pb,nzmax,iw,ierr)
    if (ierr.ne.0) call error('operat_operat_mult',ierr)
    write(*,*) 'product'
    write(*,*) size(c%a),size(c%pb)
    !call finalize(iw)
    nzmax = count(c%a.ne.0)
    write(*,*) 'nzmax=',nzmax
    c%pb(size(c%pb)) = nzmax+1
    write(*,*) 'before d'
    call new(d,c%a(1:nzmax),c%ia1(1:nzmax),c%pb,a%m,b%k)
    !call finalize(c)
    write(*,*) 'd%m',d%m,'d%k',d%k,'d%nnz',d%nnz,size(d%a),size(d%pb)
    write(*,*) 'mult done'
  end subroutine

  function braket(fi,psi)
    ! return <fi|psi>
    complex(wp) :: braket
    complex(wp), intent(in) :: fi(:),psi(:)
    braket = sum(conjg(fi)*psi)
  end function

  subroutine normalize(psi)
    complex(wp), intent(inout) :: psi(:)
    real(wp) :: tmp
    tmp = sqrt(abs(braket(psi,psi)))
    ! Check for division by zero
    if (abs(tmp) < epsi) then
      psi = 0.
    else
      psi = psi/tmp
    end if
  end subroutine

  function ket_to_operat(psi)
    ! Turns a 1d complex array psi into derived type operat
    ! psi is interpreted as a column vector
    ! i.e. no of rows = size(psi), no of columns = 1
    type(operat) :: ket_to_operat
    complex(wp), intent(in) :: psi(:)
    type(operat) :: c,d
    integer :: nzmax,ierr
    nzmax = count(psi.ne.0)
    call new(c,nzmax,size(psi)+1)
    !state_to_operat = c
    call dnscsr(size(psi),1,nzmax,psi,size(psi),c%a,c%ia1,c%pb,ierr)
    !nzmax = count(abs(c%a).ge.epsi)
    call new(d,c%a,c%ia1,c%pb,size(psi),1)
    ket_to_operat = d
  end function

  function bra_to_operat(psi)
    ! Turns a 1d complex array psi into derived type operat
    ! psi is interpreted as a row vector
    ! i.e. no of rows = 1, no of columns = size(psi)
    type(operat) :: bra_to_operat
    complex(wp), intent(in) :: psi(:)
    type(operat) :: c,d
    integer :: nzmax,ierr
    nzmax = count(psi.ne.0)
    call new(c,nzmax,1+1)
    !state_to_operat = c
    call dnscsr(1,size(psi),nzmax,psi,1,c%a,c%ia1,c%pb,ierr)
    !nzmax = count(abs(c%a).ge.epsi)
    call new(d,c%a,c%ia1,c%pb,1,size(psi))
    bra_to_operat = d
  end function

  !
  ! Misc.
  !

  subroutine densitymatrix_dense(psi,rho)
    ! Dense density matrix from pure state
    complex(wp), intent(in) :: psi(:)
    complex(wp), intent(out) :: rho(:,:)
    complex(wp), allocatable :: tmp(:,:)
    integer istat
    allocate(tmp(1,size(psi)),stat=istat)
    tmp(1,:) = psi
    rho = matmul(transpose(conjg(tmp)),tmp)
  end subroutine

  subroutine densitymatrix_sparse(psi,rho)
    ! Sparse density matrix from pure state
    complex(wp), intent(in) :: psi(:)
    type(operat), intent(out) :: rho
    ! type(operat) :: a,b
    rho = ket_to_operat(psi)*bra_to_operat(conjg(psi))
  end subroutine

  subroutine ptrace_pure(psi,rho,sel,dims)
    ! Partial trace over pure state
    ! Under construction
    ! Currently only correct for sel = (/1,2,../)
    ! i.e. no permutations
    complex(wp), intent(in) :: psi(:)
    integer, intent(in) :: sel(:),dims(:)
    complex(wp), intent(out) :: rho(:,:)
    complex(wp), allocatable :: a(:,:)
    integer :: m,n,prod_dims_sel,prod_dims_rest
    integer :: i,j,istat
    logical :: insel

    m = 1
    n = 1
    prod_dims_sel = 1
    prod_dims_rest = 1
    do i=1,size(dims)
      n=n*dims(i)
      insel = .false.
      do j=1,size(sel)
        if (i==sel(j)) then
          m=m*dims(i)
          prod_dims_sel = prod_dims_sel*dims(i)
          insel=.true.
        endif
      enddo
      if (.not.insel) then
        prod_dims_rest = prod_dims_rest*dims(i)
      endif
    enddo
    allocate(a(prod_dims_rest,prod_dims_sel),stat=istat)
    a = reshape(psi,(/prod_dims_rest,prod_dims_sel/))
    !allocate(rho(prod_dims_sel,prod_dims_sel),stat=istat)
    rho = matmul(transpose(conjg(a)),a)
  end subroutine

  !
  ! Sparse matrix routines
  !

  subroutine sparse_mv_mult(mat,x,y,ierr)
    ! y = Ax
    ! Adapted from sparse blas
    type(operat) :: mat
    complex(KIND=wp) , dimension(:), intent(in) :: x
    complex(KIND=wp) , dimension(:), intent(out) :: y
    integer, intent(out) :: ierr
    integer :: m,n,ofs,i,pntr
    ! integer :: base
    ! character :: diag,type,part
    ierr = -1
    m = size(y)
    n = size(x)
    !if ((mat%FIDA.ne.'CSR').or.(mat%M.ne.m).or.(mat%K.ne.n)) then
    !   ierr = blas_error_param
    !   return
    !end if
    !base = mat%base
    !ofs = 1 - base
    !diag = mat%diag
    !type = mat%typem
    !part = mat%part
    ofs = 0
    y = (0.0d0, 0.0d0) 
    !if (diag.eq.'U') then !process unstored diagonal
    !   if (m.eq.n) then
    !      y = x
    !   else
    !      ierr = blas_error_param
    !      return
    !   end if
    !end if
    !if ((type.eq.'S').and.(.not.(part.eq.'B')).and.(m.eq.n)) then 
    !   if (part.eq.'U') then
    !      do i = 1, mat%M
    !         pntr = mat%pb(i)
    !         do while(pntr.lt.mat%pe(i))
    !            if(i.eq.mat%IA1(pntr + ofs) + ofs) then
    !               y(i) = y(i) &
    !            + mat%A(pntr + ofs) * x(mat%IA1(pntr + ofs ) + ofs) 
    !            else if (i.lt.mat%IA1(pntr + ofs) + ofs) then
    !               y(i) = y(i) &
    !            + mat%A(pntr + ofs) * x(mat%IA1(pntr + ofs ) + ofs) 
    !               y(mat%IA1(pntr + ofs) + ofs) =  &
    !        y(mat%IA1(pntr + ofs ) + ofs) + mat%A(pntr + ofs) * x(i) 
    !            end if
    !            pntr = pntr + 1
    !         end do
    !      end do
    !   else
    !      do i = 1, mat%M
    !         pntr = mat%pb(i)
    !         do while(pntr.lt.mat%pe(i))
    !            if(i.eq.mat%IA1(pntr + ofs) + ofs) then
    !               y(i) = y(i) &
    !            + mat%A(pntr + ofs) * x(mat%IA1(pntr + ofs ) + ofs) 
    !            else if (i.gt.mat%IA1(pntr + ofs) + ofs) then
    !               y(i) = y(i) &
    !            + mat%A(pntr + ofs) * x(mat%IA1(pntr + ofs ) + ofs) 
    !               y(mat%IA1(pntr + ofs) + ofs) = &
    !        y(mat%IA1(pntr + ofs ) + ofs) + mat%A(pntr + ofs) * x(i) 
    !            end if
    !            pntr = pntr + 1
    !         end do
    !      end do
    !   end if
    !   ierr = 0
    !else if((type.eq.'H').and.(.not.(part.eq.'B')).and.(m.eq.n)) then 
    !   if (part.eq.'U') then
    !      do i = 1, mat%M
    !         pntr = mat%pb(i)
    !         do while(pntr.lt.mat%pe(i))
    !            if(i.eq.mat%IA1(pntr + ofs) + ofs) then
    !               y(i) = y(i) &
    !            + mat%A(pntr + ofs) * x(mat%IA1(pntr + ofs ) + ofs) 
    !            else if (i.lt.mat%IA1(pntr + ofs) + ofs) then
    !               y(i) = y(i) &
    !            + mat%A(pntr + ofs) * x(mat%IA1(pntr + ofs ) + ofs) 
    !              y(mat%IA1(pntr+ofs)+ofs)=y(mat%IA1(pntr+ofs)+ofs) &
    !                     + conjg (mat%A(pntr + ofs)) * x(i) 
    !            end if
    !            pntr = pntr + 1
    !         end do
    !      end do
    !   else
    !      do i = 1, mat%M
    !         pntr = mat%pb(i)
    !         do while(pntr.lt.mat%pe(i))
    !            if(i.eq.mat%IA1(pntr + ofs) + ofs) then
    !               y(i) = y(i) &
    !            + mat%A(pntr + ofs) * x(mat%IA1(pntr + ofs ) + ofs) 
    !            else if (i.gt.mat%IA1(pntr + ofs) + ofs) then
    !               y(i) = y(i) &
    !            + mat%A(pntr + ofs) * x(mat%IA1(pntr + ofs ) + ofs) 
    !             y(mat%IA1(pntr+ofs)+ofs)=y(mat%IA1(pntr+ofs)+ofs) &
    !                     + conjg (mat%A(pntr + ofs)) * x(i) 
    !            end if
    !            pntr = pntr + 1
    !         end do
    !      end do
    !   end if
    !   ierr = 0
    !else
       do i = 1, mat%M
          pntr = mat%pb(i)
          !do while(pntr.lt.mat%pe(i))
          do while(pntr.lt.mat%pb(i+1))
            y(i) = y(i) &
                + mat%A(pntr + ofs) * x(mat%IA1(pntr + ofs ) + ofs) 
             pntr = pntr + 1
          end do
       end do
       ierr = 0
    !end if
  end subroutine


!  subroutine amux ( n, x, y, a, ja, ia )
!  ! Aadapted from sparsekit
!
!  !*****************************************************************************80
!  !
!  !! AMUX multiplies a CSR matrix A times a vector.
!  !
!  !  Discussion:
!  !
!  !    This routine multiplies a matrix by a vector using the dot product form.
!  !    Matrix A is stored in compressed sparse row storage.
!  !
!  !  Modified:
!  !
!  !    07 January 2004
!  !
!  !  Author:
!  !
!  !    Youcef Saad
!  !
!  !  Parameters:
!  !
!  !    Input, integer N, the row dimension of the matrix.
!  !
!  !    Input, real X(*), and array of length equal to the column dimension 
!  !    of A.
!  !
!  !    Input, real A(*), integer JA(*), IA(NROW+1), the matrix in CSR
!  !    Compressed Sparse Row format.
!  !
!  !    Output, real Y(N), the product A * X.
!  !
!    implicit none
!
!    integer n
!
!    complex ( kind = wp ) a(*)
!    integer i
!    integer ia(*)
!    integer ja(*)
!    integer k
!    complex ( kind = wp ) t
!    complex ( kind = wp ) x(*)
!    complex ( kind = wp ) y(n)
!
!    do i = 1, n
!  !
!  !  Compute the inner product of row I with vector X.
!  !
!      t = (0.0,0.0)
!      do k = ia(i), ia(i+1)-1
!        t = t + a(k) * x(ja(k))
!      end do
!
!      y(i) = t
!
!    end do
!
!    return
!  end
!

subroutine aplb ( nrow, ncol, job, a, ja, ia, b, jb, ib, c, jc, ic, nzmax, &
  iw, ierr )

! Adapted from sparsekit

!*****************************************************************************80
!
!! APLB performs the CSR matrix sum C = A + B.
!
!  Modified:
!
!    07 January 2004
!
!  Author:
!
!    Youcef Saad
!
!  Parameters:
!
!    Input, integer NROW, the row dimension of A and B.
!
!    Input, integer NCOL, the column dimension of A and B.
!
!    Input, integer JOB.  When JOB = 0, only the structure
!    (i.e. the arrays jc, ic) is computed and the
!    real values are ignored.
!
!    Input, real A(*), integer JA(*), IA(NROW+1), the matrix in CSR
!    Compressed Sparse Row format.
!
! b,
! jb,
! ib      =  Matrix B in compressed sparse row format.
!
! nzmax      = integer. The  length of the arrays c and jc.
!         amub will stop if the result matrix C  has a number
!         of elements that exceeds exceeds nzmax. See ierr.
!
! on return:
!
! c,
! jc,
! ic      = resulting matrix C in compressed sparse row sparse format.
!
! ierr      = integer. serving as error message.
!         ierr = 0 means normal return,
!         ierr > 0 means that amub stopped while computing the
!         i-th row  of C with i = ierr, because the number
!         of elements in C exceeds nzmax.
!
! work arrays:
!
! iw      = integer work array of length equal to the number of
!         columns in A.
!
  implicit none

  integer ncol
  integer nrow

  complex ( kind = wp ) a(*)
  complex ( kind = wp ) b(*)
  complex ( kind = wp ) c(*)
  integer ia(nrow+1)
  integer ib(nrow+1)
  integer ic(nrow+1)
  integer ierr
  integer ii
  integer iw(ncol)
  integer ja(*)
  integer jb(*)
  integer jc(*)
  integer jcol
  integer job
  integer jpos
  integer k
  integer ka
  integer kb
  integer len
  integer nzmax
  logical values

  values = ( job /= 0 )
  ierr = 0
  len = 0
  ic(1) = 1
  iw(1:ncol) = 0

  do ii = 1, nrow
!
!  Row I.
!
     do ka = ia(ii), ia(ii+1)-1

        len = len + 1
        jcol = ja(ka)

        if ( nzmax < len ) then
          ierr = ii
          return
        end if

        jc(len) = jcol
        if ( values ) then
          c(len) = a(ka)
        end if
        iw(jcol) = len
     end do

     do kb = ib(ii), ib(ii+1)-1

        jcol = jb(kb)
        jpos = iw(jcol)

        if ( jpos == 0 ) then

           len = len + 1

           if ( nzmax < len ) then
             ierr = ii
             return
           end if

           jc(len) = jcol
           if ( values ) then
             c(len) = b(kb)
           end if
           iw(jcol)= len
        else
           if ( values ) then
             c(jpos) = c(jpos) + b(kb)
           end if
        end if

     end do

     do k = ic(ii), len
       iw(jc(k)) = 0
     end do

     ic(ii+1) = len+1
  end do

  return
end subroutine
subroutine amub ( nrow, ncol, job, a, ja, ia, b, jb, ib, c, jc, ic, nzmax, &
  iw, ierr )
  ! Aadapted from sparsekit

!*****************************************************************************80
!
!! AMUB performs the matrix product C = A * B.
!
!  Discussion:
!
!    The column dimension of B is not needed.
!
!  Modified:
!
!    08 January 2004
!
!  Author:
!
!    Youcef Saad
!
!  Parameters:
!
!    Input, integer NROW, the row dimension of the matrix.
!
!    Input, integer NCOL, the column dimension of the matrix.
!
!    Input, integer JOB, job indicator.  When JOB = 0, only the structure
!    is computed, that is, the arrays JC and IC, but the real values
!    are ignored.
!
!    Input, real A(*), integer JA(*), IA(NROW+1), the matrix in CSR
!    Compressed Sparse Row format.
!
!    Input, b, jb, ib, matrix B in compressed sparse row format.
!
!    Input, integer NZMAX, the length of the arrays c and jc.
!    The routine will stop if the result matrix C  has a number
!    of elements that exceeds exceeds NZMAX.
!
! on return:
!
! c,
! jc,
! ic    = resulting matrix C in compressed sparse row sparse format.
!
! ierr      = integer. serving as error message.
!         ierr = 0 means normal return,
!         ierr > 0 means that amub stopped while computing the
!         i-th row  of C with i = ierr, because the number
!         of elements in C exceeds nzmax.
!
! work arrays:
!
!  iw      = integer work array of length equal to the number of
!         columns in A.
!
  implicit none

  integer ncol
  integer nrow
  integer nzmax

  complex ( kind = wp ) a(*)
  complex ( kind = wp ) b(*)
  complex ( kind = wp ) c(nzmax)
  integer ia(nrow+1)
  integer ib(ncol+1)
  integer ic(ncol+1)
  integer ierr
  integer ii
  integer iw(ncol)
  integer ja(*)
  integer jb(*)
  integer jc(nzmax)
  integer jcol
  integer jj
  integer job
  integer jpos
  integer k
  integer ka
  integer kb
  integer len
  complex ( kind = wp ) scal
  logical values

  values = ( job /= 0 )
  len = 0
  ic(1) = 1
  ierr = 0
!
!  Initialize IW.
!
  iw(1:ncol) = 0

  do ii = 1, nrow
!
!  Row I.
!
    do ka = ia(ii), ia(ii+1)-1

      if ( values ) then
        scal = a(ka)
      end if

      jj = ja(ka)

      do kb = ib(jj), ib(jj+1)-1

           jcol = jb(kb)
           jpos = iw(jcol)

           if ( jpos == 0 ) then
              len = len + 1
              if ( nzmax < len ) then
                 ierr = ii
                 return
              end if
              jc(len) = jcol
              iw(jcol)= len
              if ( values ) then
                c(len) = scal * b(kb)
              end if
           else
              if ( values ) then
                c(jpos) = c(jpos) + scal * b(kb)
              end if
           end if

         end do

    end do

    do k = ic(ii), len
      iw(jc(k)) = 0
    end do

    ic(ii+1) = len + 1

  end do

  return
end subroutine


subroutine dnscsr ( nrow, ncol, nzmax, dns, ndns, a, ja, ia, ierr )

! Adapted from Sparsekit

!*****************************************************************************80
!
!! DNSCSR converts Dense to Compressed Row Sparse format.
!
!  Discussion:
!
!    This routine converts a densely stored matrix into a row orientied
!    compactly sparse matrix.  It is the reverse of CSRDNS.
!
!    This routine does not check whether an element is small.  It considers 
!    that A(I,J) is zero only if it is exactly equal to zero.
!
!  Modified:
!
!    07 January 2004
!
!  Author:
!
!    Youcef Saad
!
!  Parameters:
!
!    Input, integer NROW, the row dimension of the matrix.
!
!    Input, integer NCOL, the column dimension of the matrix.
!
!    Input, integer NZMAX, the maximum number of nonzero elements 
!    allowed.  This should be set to be the lengths of the arrays A and JA.
!
!    Input, real DNS(NDNS,NCOL), an NROW by NCOL dense matrix.
!
!    Input, integer NDNS, the first dimension of DNS, which must be
!    at least NROW.
!
!    Output, real A(*), integer JA(*), IA(NROW+1), the matrix in CSR
!    Compressed Sparse Row format.
!
!    Output, integer IERR, error indicator.
!    0 means normal return;
!    I, means that the the code stopped while processing row I, because
!       there was no space left in A and JA, as defined by NZMAX.
!
  implicit none

  integer ncol
  integer ndns
  integer nrow

  complex ( kind = wp ) a(*)
  complex ( kind = wp ) dns(ndns,ncol)
  integer i
  integer ia(nrow+1)
  integer ierr
  integer j
  integer ja(*)
  integer next
  integer nzmax

  ierr = 0
  next = 1
  ia(1) = 1

  do i = 1, nrow

    do j = 1, ncol

      if ( dns(i,j) /= 0.0D+00 ) then

        if ( nzmax < next ) then
          ierr = i
          return
        end if

        ja(next) = j
        a(next) = dns(i,j)
        next = next + 1

      end if

    end do

    ia(i+1) = next

  end do

  return
end subroutine

end module
