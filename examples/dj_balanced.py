from state import State

state = State(n_qubits=2)
state.h(0)
state.x(1)
state.h(1)
## balanced oracle
state.cx(0,1)
state.x(1)
##
state.h(0)
state.measure(0)
