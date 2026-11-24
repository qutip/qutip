import mpi4py, mpi4py.MPI as MPI
print('mpi4py:', mpi4py.__version__)
print('MPI std:', MPI.Get_version())
print('Library:', MPI.Get_library_version())
print('Config:', mpi4py.get_config())
