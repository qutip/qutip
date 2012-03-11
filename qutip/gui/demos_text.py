from scipy import arange
#basic demos
basic_labels=["Schrodingers Cat","Q-function","Qobj Eigenvalues/Eigenvectors","blank","blank","blank","blank"]

basic_desc=["Schrodingers Cat state from \nsuperposition of two coherent states.",
                            "Q-function from superposition \nof two coherent states.",
                            "Eigenvalues/Eigenvectors of cavity-qubit system \nin strong-coupling regime.",
                            "Bloch Sphere","blank","blank","blank"]

basic_nums=arange(1,len(basic_labels)+1) #does not start at zero so commandline output numbers match (0=quit in commandline)

#master equation demos
master_labels=["blank","blank","blank","blank","blank","blank","blank"]
master_desc=["blank","blank","blank","blank","blank","blank","blank"]
master_nums=10+arange(len(master_labels))

monte_labels=["blank","blank","blank","blank","blank","blank","blank"]
monte_desc=["blank","blank","blank","blank","blank","blank","blank"]
monte_nums=20+arange(len(monte_labels))

redfield_labels=["blank","blank","blank","blank","blank","blank","blank"]
redfield_desc=["blank","blank","blank","blank","blank","blank","blank"]
redfield_nums=30+arange(len(redfield_labels))

td_labels=["blank","blank","blank","blank","blank","blank","blank"]
td_desc=["blank","blank","blank","blank","blank","blank","blank"]
td_nums=40+arange(len(td_labels))

tab_labels=['Basic Operations','Master Equation','Monte Carlo','Bloch-Redfield','Time-Dependent']
button_labels=[basic_labels,master_labels,monte_labels,redfield_labels,td_labels]
button_desc=[basic_desc,master_desc,monte_desc,redfield_desc,td_desc]
button_nums=[basic_nums,master_nums,monte_nums,redfield_nums,td_nums]