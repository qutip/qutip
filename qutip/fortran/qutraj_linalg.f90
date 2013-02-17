module qutraj_linalg
  !
  ! This module depends on LAPACK
  !

  use qutraj_precision
  use qutraj_general
  use qutraj_hilbert

  implicit none

  contains

  subroutine eigenvalues(rho,eig,n)
    ! Eigenvalues of dense hermitian matrix rho
    complex(wp), intent(in) :: rho(:,:)
    integer, intent(in) :: n
    real(wp), intent(out) :: eig(n)
    double complex :: ap(n*(n+1)/2), z(1,1),work(2*n-1)
    double precision :: eig_dp(n),rwork(3*n-2)
    integer info,i,j
    do i=1,n
      do j=i,n
        ap(i+(j-1)*j/2) = rho(i,j)
      enddo
    enddo
    call zhpev('N','U',n,ap,eig_dp,z,1,work,rwork,info)
    eig = eig_dp
  end subroutine

  subroutine entropy(rho,S)
    ! Calculate entropy for dense density matrix
    complex(wp), intent(in) :: rho(:,:)
    real(wp), intent(out) :: S
    real(wp), dimension(2) :: eig_r
    integer :: i
    call eigenvalues(rho,eig_r,size(rho,1))
    S = 0
    do i=1,size(eig_r)
      ! Rule: 0 log(0) = 0
      if (eig_r(i) < -epsi) &
        write(*,*) "entropy: negative eigenvalue!", eig_r(i)
      if (abs(eig_r(i)) > epsi) then
        S = S -eig_r(i)*log(eig_r(i))/log(2.)
      endif
    enddo
  end subroutine


end module
