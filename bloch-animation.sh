#
# create a movie from a sequence of bloch sphere figures
#
mencoder mf://bloch-plots/*.jpg -mf w=800:h=600:fps=25:type=jpg -ovc lavc \
    -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o bloch-animation.mpeg
