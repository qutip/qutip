module qutraj_precision
  ! Module for setting working precision

  implicit none

  !integer, parameter :: sp = kind(1.0e0) ! single precision
  !integer, parameter :: dp = kind(1.0d0) ! double precision
  integer, parameter :: sp = selected_real_kind(6,37) ! single precision
  integer, parameter :: dp = selected_real_kind(15,307) ! double precision
  integer, parameter :: wp = dp ! working precision

  ! small and large number
  real, parameter :: epsi=5*epsilon(1.0)
  real, parameter :: huge1=0.2*huge(1.0)

end module
