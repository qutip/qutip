module qutraj_general
  !
  ! Global constants and general purpose subroutines
  !

  use qutraj_precision

  implicit none

  !
  ! Constants
  !

  ! error params
  integer, parameter :: blas_error_param    = -23

  ! imaginary unit
  complex(wp), parameter :: ii = (0._wp,1._wp)

  !
  ! Interfaces
  !

  interface new
    module procedure int_array_init
    module procedure int_array_init2
    module procedure wp_array_init
    module procedure wp_array_init2
  end interface

  interface finalize
    module procedure int_array_finalize
    module procedure wp_array_finalize
  end interface

  contains

  !
  ! Initializers and finalizers
  !

  subroutine int_array_init(this,n)
    integer, allocatable, intent(inout) :: this(:)
    integer, intent(in) :: n
    integer :: istat
    if (allocated(this)) then
      deallocate(this,stat=istat)
    endif
    allocate(this(n),stat=istat)
    if (istat.ne.0) then
      call fatal_error("int_array_init: could not allocate.",istat)
    endif
  end subroutine

  subroutine int_array_init2(this,val)
    integer, allocatable, intent(inout) :: this(:)
    integer, intent(in), dimension(:) :: val
    call int_array_init(this,size(val))
    this = val
  end subroutine

  subroutine wp_array_init(this,n)
    real(wp), allocatable, intent(inout) :: this(:)
    integer, intent(in) :: n
    integer :: istat
    if (allocated(this)) then
      deallocate(this,stat=istat)
    endif
    allocate(this(n),stat=istat)
    if (istat.ne.0) then
      call fatal_error("sp_array_init: could not allocate.",istat)
    endif
  end subroutine

  subroutine wp_array_init2(this,val)
    real(wp), allocatable, intent(inout) :: this(:)
    real(wp), intent(in), dimension(:) :: val
    call wp_array_init(this,size(val))
    this = val
  end subroutine

  subroutine int_array_finalize(this)
    integer, allocatable, intent(inout) :: this(:)
    integer :: istat=0
    if (allocated(this)) then
      deallocate(this,stat=istat)
    endif
    if (istat.ne.0) then
      call error("int_array_finalize: could not deallocate.",istat)
    endif
  end subroutine

  subroutine wp_array_finalize(this)
    real(wp), allocatable, intent(inout) :: this(:)
    integer :: istat=0
    if (allocated(this)) then
      deallocate(this,stat=istat)
    endif
    if (istat.ne.0) then
      call error("wp_array_finalize: could not deallocate.",istat)
    endif
  end subroutine

  !
  ! Error handling
  !

  subroutine error(errormsg,ierror)
    character(len=*), intent(in), optional :: errormsg
    integer, intent(in), optional :: ierror
    if (present(errormsg)) then
      write(*,*) 'error: ',errormsg
    endif
    if (present(ierror)) then
      write(*,*) 'error flag=',ierror
    endif
  end subroutine

  subroutine fatal_error(errormsg,ierror)
    character(len=*), intent(in), optional :: errormsg
    integer, intent(in), optional :: ierror
    if (present(errormsg)) then
      write(*,*) 'fatal error: ',errormsg
    endif
    if (present(ierror)) then
      write(*,*) 'error flag=',ierror
    endif
    write(*,*) 'halting'
    stop 1
  end subroutine

end module
