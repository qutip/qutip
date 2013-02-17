module qutraj_linalg
  !
  ! Dummy module
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
    call error('eigenvalues: subroutine not available.')
  end subroutine

  subroutine entropy(rho,S)
    ! Calculate entropy for dense density matrix
    complex(wp), intent(in) :: rho(:,:)
    real(wp), intent(out) :: S
    call error('eigenvalues: subroutine not available.')
  end subroutine

end module
