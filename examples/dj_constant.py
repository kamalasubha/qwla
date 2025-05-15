from state import State

state = State(n_qubits=2)
state.h(0)
state.x(1)
state.h(1)
## constant oracle
state.x(1)
##
state.h(0)
state.measure(0)
