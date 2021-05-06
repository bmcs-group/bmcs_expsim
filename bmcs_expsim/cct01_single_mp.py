# Single material point simulation with MS1
import matplotlib.pylab as plt
from bmcs_matmod.ms1.ms1 import MS13D
import bmcs_matmod.ms1.concrete_material_db as mp_db
import numpy as np


tmodel = MS13D(**mp_db.C40MS1)



DELTA = np.identity(3)

EPS = np.zeros((3, 3, 3), dtype='f')
EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1

DD = np.hstack([DELTA, np.zeros_like(DELTA)])
EEPS = np.hstack([np.zeros_like(EPS), EPS])

GAMMA = np.einsum(
    'ik,jk->kij', DD, DD
) + np.einsum(
    'ikj->kij', np.fabs(EEPS)
)


def get_eps_ab(eps_O): return np.einsum(
    'Oab,...O->...ab', GAMMA, eps_O
)[np.newaxis, ...]


GAMMA_inv = np.einsum(
    'aO,bO->Oab', DD, DD
) + 0.5 * np.einsum(
    'aOb->Oab', np.fabs(EEPS)
)


def get_sig_O(sig_ab): return np.einsum(
    'Oab,...ab->...O', GAMMA_inv, sig_ab
)[0, ...]


GG = np.einsum(
    'Oab,Pcd->OPabcd', GAMMA_inv, GAMMA_inv
)


def get_K_OP(D_abcd):
    return np.einsum(
        'OPabcd,abcd->OP', GG, D_abcd
    )


eps_ab = np.zeros((3, 3), dtype=np.float_)
sig_range = []
n_O = 6
F_ext = np.zeros((n_O,), np.float_)
F_O = np.zeros((n_O,), np.float_)
U_P = np.zeros((n_O,), np.float_)
U_k_O = np.zeros((n_O,), dtype=np.float_)
state_arrays = {name: np.zeros(shape_, dtype=np.float_) for
                name, shape_ in tmodel.state_var_shapes.items()}
CONTROL = 0
FREE = slice(1, None)  # This means all except the first index, i.e. [1:]

for i in range(1, len(eps_range)):

    k = 0
    k_max, R_acc = 1000, 1e-3
    F_ext = F_O
    delta_U = eps_range[i] - eps_range[i - 1]
    # Equilibrium iteration loop
    while k < k_max:
        # Transform the primary vector to field
        eps_ab = get_eps_ab(U_k_O).reshape(3, 3)
        # Stress and material stiffness
        sig_ab, D_abcd = tmodel.get_corr_pred(
            eps_ab, 0, **state_arrays
        )
        F_O = get_sig_O(sig_ab.reshape(1, 3, 3)).reshape(6, )
        # System matrix
        K_OP = get_K_OP(D_abcd)
        # Beta = get_K_OP(beta_Emabcd)
        # Get the balancing forces - NOTE - for more displacements
        # this should be an assembly operator.
        # KU remains a 2-d array so we have to make it a vector
        KU = K_OP[:, CONTROL] * delta_U
        # Residuum
        R_O = F_ext - F_O - KU
        # Convergence criterion
        R_norm = np.linalg.norm(R_O[FREE])
        if R_norm < R_acc:
            # Convergence reached
            break
        # Next iteration -
        delta_U_O = np.linalg.solve(K_OP[FREE, FREE], R_O[FREE])
        # Update total displacement
        U_k_O[FREE] += delta_U_O
        # Update control displacement
        U_k_O[CONTROL] += delta_U
        # Note - control displacement nonzero only in the first iteration.
        delta_U = 0
        k += 1

    else:
        print('no convergence')
        break
    # Update internal variables here
    sig_range.append(sig_ab[0, 0])

# %% md

## Solve possible issue of internal variables update
if __name__ == "__main__":

    F = []