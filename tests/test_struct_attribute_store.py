import numpy as np

import numeta as nm


def test_struct_attribute_read_and_store(backend):
    @nm.jit(backend=backend)
    def kernel(particles):
        particles[0].mass = particles[1].mass + 2.0

    particles = np.zeros(2, dtype=np.dtype([("mass", np.float64)], align=True))
    particles[1]["mass"] = 3
    kernel(particles)
    assert particles[0]["mass"] == 5
