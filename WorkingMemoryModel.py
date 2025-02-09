# Imports
import numpy as np
import nengo.spa as spa
import matplotlib.pyplot as plt

import nengo

# Seed for Reproducability
seed = 0
np.random.seed(seed)

# Initialize Vocabulary
vocab = spa.Vocabulary(dimensions=32)

vocab.add("CAT", vocab.create_pointer())  
vocab.add("DOG", vocab.create_pointer())  
vocab.add("ZERO", np.zeros(32))  

# Input Function
def input_func(t):
    if 0 <= t < 0.4:
        return vocab["CAT"].v  
    elif 0.4 <= t < 0.8:
        return vocab["DOG"].v  
    elif 0.8 <= t < 1.2:
        return vocab["CAT"].v 
    elif 1.2 <= t < 1.6:
        return vocab["DOG"].v  
    else:
        return vocab["ZERO"].v  



# Create Model
with spa.SPA(seed = seed) as model:
    # Visual Input
    model.visual_input = spa.State(dimensions=32)

    # Working Memory with Feedback
    model.working_memory = spa.State(dimensions=32, feedback=1)

    # Input Node
    stim = nengo.Node(input_func)

    # Connect Input Node with Visual Input
    nengo.Connection(stim, model.visual_input.input)

    # Connect Visual Input to Working Memory
    nengo.Connection(model.visual_input.output, model.working_memory.input, transform=0.1)

    # Probes
    visual_probe = nengo.Probe(model.visual_input.output, synapse=0.03)
    memory_probe = nengo.Probe(model.working_memory.output, synapse=0.03)


# Run Simulation
with nengo.Simulator(model) as sim:
    sim.run(20.0)

# Plotting Similarity
plt.figure(figsize=(12, 6))
similarity_visual = spa.similarity(sim.data[visual_probe], vocab)
plt.plot(sim.trange(), similarity_visual[:, 0], label="CAT (Visual Input)")
plt.plot(sim.trange(), similarity_visual[:, 1], label="DOG (Visual Input)")
plt.plot(sim.trange(), similarity_visual[:, 2], label="ZERO (Visual Input)")
plt.xlabel("Time (s)")
plt.ylabel("Similarity")
plt.title("Similarity Plot for Visual Input Over 20s")
plt.legend(loc="upper right")
plt.grid()
plt.show()

# Plotting Working Memory
plt.figure(figsize=(12, 6))
similarity_memory = spa.similarity(sim.data[memory_probe], vocab)
plt.plot(sim.trange(), similarity_memory[:, 0], label="CAT (Working Memory)")
plt.plot(sim.trange(), similarity_memory[:, 1], label="DOG (Working Memory)")
plt.plot(sim.trange(), similarity_memory[:, 2], label="ZERO (Working Memory)")
plt.xlabel("Time (s)")
plt.ylabel("Similarity")
plt.title("Similarity Plot for Working Memory Output Over 20s")
plt.legend(loc="upper right")
plt.grid()
plt.show()