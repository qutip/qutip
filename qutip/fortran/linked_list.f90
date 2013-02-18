module linked_list
  !
  ! Linked list module
  !

  use qutraj_precision

  implicit none

  type llnode_real
    type(llnode_real), pointer :: next=>null()
    real(wp) :: a
  end type

  type llnode_int
    type(llnode_int), pointer :: next=>null()
    integer :: a
  end type

  type linkedlist_real
    type(llnode_real), pointer :: head=>null(), tail=>null()
    integer :: nelements = 0
  end type

  type linkedlist_int
    type(llnode_int), pointer :: head=>null(), tail=>null()
    integer :: nelements = 0
  end type

  interface new
    module procedure init_node_real
    module procedure init_node_int
  end interface

  interface finalize
    module procedure finalize_ll_real
    module procedure finalize_ll_int
  end interface

  interface append
    module procedure ll_append_real
    module procedure ll_append_int
  end interface

  interface ll_to_array
    module procedure ll_to_array_real
    module procedure ll_to_array_int
  end interface

  contains

  subroutine init_node_real(node,a)
    type(llnode_real), pointer, intent(inout) :: node
    real(wp) :: a
    allocate(node)
    node%a=a
    node%next=>null()
  end subroutine

  subroutine init_node_int(node,a)
    type(llnode_int), pointer, intent(inout) :: node
    integer :: a
    allocate(node)
    node%a=a
    node%next=>null()
  end subroutine

  subroutine ll_append_real(list, a)
    !Add a node to the end of the list.
    type(linkedlist_real), intent(inout) :: list
    real(wp), intent(in) :: a
    type(llnode_real), pointer :: node
    call new(node,a)
    if (associated(list%head)) then
      list%tail%next => node  
      node%next => null()  
      list%tail => node  
    else
      list%head => node    
      list%tail => node  
      list%tail%next => null()  
    end if
    list%nelements = list%nelements+1
  end subroutine

  subroutine ll_append_int(list, a)
    !Add a node to the end of the list.
    type(linkedlist_int), intent(inout) :: list
    integer, intent(in) :: a
    type(llnode_int), pointer :: node
    call new(node,a)
    if (associated(list%head)) then
      list%tail%next => node  
      node%next => null()  
      list%tail => node  
    else
      list%head => node    
      list%tail => node  
      list%tail%next => null()  
    end if
    list%nelements = list%nelements+1
  end subroutine

  subroutine ll_to_array_real(list, table)
    ! Makes an array out of the list
    ! while deleting the list nodes!
    type(linkedlist_real), intent(inout) :: list
    real(wp), allocatable, intent(out) :: table(:)
    type(llnode_real), pointer :: move, tmp
    integer :: i
    ! Check if empty.
    if (.not. associated(list%head)) then
      return
    else
      ! Allocate table
      allocate(table(list%nelements))
      ! Load the table with the list.
      move=>list%head
      do i=1, list%nelements
        table(i)=move%a
        if (associated(move%next)) then
          tmp=>move
          move=>move%next
          deallocate(tmp)
          nullify(tmp)
        endif
      end do
      list%head => null()
      list%tail => null()
    end if
  end subroutine

  subroutine ll_to_array_int(list, table)
    ! Makes an array out of the list
    ! while deleting the list nodes!
    type(linkedlist_int), intent(inout) :: list
    integer, allocatable, intent(out) :: table(:)
    type(llnode_int), pointer :: move, tmp
    integer :: i
    ! Check if empty.
    if (.not. associated(list%head)) then
      return
    else
      ! Allocate table
      allocate(table(list%nelements))
      ! Load the table with the list.
      move=>list%head
      do i=1, list%nelements
        table(i)=move%a
        if (associated(move%next)) then
          tmp=>move
          move=>move%next
          deallocate(tmp)
          nullify(tmp)
        endif
      end do
      list%head => null()
      list%tail => null()
    end if
  end subroutine

  !Delete all elements in a list.  Leaves the list initialized.
  subroutine finalize_ll_real(list)
  implicit none
    type(linkedlist_real), intent(inout) :: list
    type(llnode_real), pointer :: move
    do
      !Check if list empty.
      if (.not. associated(list%head)) then
        exit
      else
        !Check if more than 1 node.
        if (associated(list%head%next)) then !more than one node.
          move => list%head
          list%head => list%head%next
          move%next => null()
        else
          move => list%head
          list%head => null()
          list%tail => null()
        end if
        !call ll_del_first(list,move)
        deallocate(move)
        nullify(move)
      end if      
    end do
  end subroutine

  subroutine finalize_ll_int(list)
  implicit none
    type(linkedlist_int), intent(inout) :: list
    type(llnode_int), pointer :: move
    do
      !Check if list empty.
      if (.not. associated(list%head)) then
        exit
      else
        !Check if more than 1 node.
        if (associated(list%head%next)) then !more than one node.
          move => list%head
          list%head => list%head%next
          move%next => null()
        else
          move => list%head
          list%head => null()
          list%tail => null()
        end if
        !call ll_del_first(list,move)
        deallocate(move)
        nullify(move)
      end if      
    end do
  end subroutine

end module linked_list


























