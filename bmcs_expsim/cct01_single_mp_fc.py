# Single material point simulation with MS1
import matplotlib.pylab as plt
from bmcs_matmod.ms1.ms1 import MS13D
import bmcs_matmod.ms1.concrete_material_db as mp_db
import numpy as np
import matplotlib
import pandas as pd
import time
import copy


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




def get_UF_t(F, n_t, load, S_max1, S_max2, S_min1, tmodel, loading_scenario):

    eps_ab = np.zeros((3, 3), dtype=np.float_)
    U_t_list, F_t_list = [] , []
    n_O = 6
    F_ext = np.zeros((n_O,), np.float_)
    F_O = np.zeros((n_O,), np.float_)
    U_P = np.zeros((n_O,), np.float_)
    U_k_O = np.zeros((n_O,), dtype=np.float_)
    state_arrays = {name: np.zeros(shape_, dtype=np.float_) for
                    name, shape_ in tmodel.state_var_shapes.items()}

    copy_state_arrays = {name: np.zeros(shape_, dtype=np.float_) for
                    name, shape_ in tmodel.state_var_shapes.items()}

    t_aux, t_n1, t_max, t_step = 0, 0, len(F), 1 / n_t
    k_max, R_acc = 1000, 1e-3

    while t_n1 <= t_max - 1:

        k = 0

        F_ext[0] = F[t_n1]

        #print(copy_state_arrays["omega_N_Emn"])

        # Equilibrium iteration loop
        while k < k_max:
            # Transform the primary vector to field
            eps_ab = get_eps_ab(U_k_O).reshape(3, 3)
            # Stress and material stiffness
            sig_ab, D_abcd = tmodel.get_corr_pred(
                eps_ab, 0, **copy_state_arrays
            )
            F_O = get_sig_O(sig_ab.reshape(1, 3, 3)).reshape(6, )
            # System matrix
            K_OP = get_K_OP(D_abcd)
            # Residuum
            R_O = F_ext - F_O
            # Convergence criterion
            R_norm = np.linalg.norm(R_O)
            # Next iteration -
            delta_U_O = np.linalg.solve(K_OP, R_O)
            # Update total displacement
            U_k_O += delta_U_O
            if R_norm < R_acc:
                # Convergence reached
                break
            # Update control displacement
            # Note - control displacement nonzero only in the first iteration.
            k += 1
        else:
            print('no convergence')
            break
        # Update internal variables here
        copy_state_arrays = copy.copy(state_arrays)
        if loading_scenario == 'constant':
            # Saving data just at min and max levels
            if F[t_n1] == 0 or F[t_n1] == S_max1 * load or F[t_n1] == S_min1 * load:
                U_t_list.append(np.copy(U_k_O))
                F_t_list.append(np.copy(F_O))
                t_aux += 1

        if loading_scenario == 'order':
            # Saving data just at min and max levels
            if F[t_n1] == 0 or F[t_n1] == S_max1 * load or F[t_n1] == S_max2 * load or F[t_n1] == S_min1 * load:
                U_t_list.append(np.copy(U_k_O))
                F_t_list.append(np.copy(F_O))
                t_aux += 1

        t_n1 += 1
    U_t, F_t = np.array(U_t_list), np.array(F_t_list)
    return U_t, F_t, t_n1 / t_max, t_aux


if __name__ == "__main__":


    concrete_type = 0  # 0:C40MA, 1:C80MA, 2:120MA, 3:Tensile, 4:Compressive, 5:Biaxial, 6:Confinement, 7:Sensitivity

    tmodel = MS13D(**mp_db.C40MS1)

    loading_scenario = 'constant'  # constant, order, increasing

    t_steps_cycle = 100
    n_mp = 28

    S_max1 = 0.9  # maximum loading level
    S_min1 = 0.05  # minimum loading level
    n_cycles1 = 1000  # number of applied cycles

    # For sequence order effect

    eta1 = 0.15  # fatigue life fraction first level
    cycles1 = 20  # fatigue life first level

    S_max2 = 0.85  # maximum loading level second level
    cycles2 = 221  # fatigue life second level
    n_cycles2 = int(1e3 - np.floor(eta1 * cycles1))  # number of applied cycles second level

    load_options = [-60.58945931264715, -93.73515724992052, -121.1045747178644, 3.5554010452401035, -42.67767356282073,
                    -44.98639244345037, -10.,
                    -47.57477583679554]  # -48.13556186920362,-48.138755539378714, -47.57477583679554

    load = load_options[concrete_type]

    # LOADING SCENARIOS

    first_load = np.concatenate((np.linspace(0, load * S_max1, t_steps_cycle), np.linspace(
        load * S_max1, load * S_min1, t_steps_cycle)[1:]))

    if loading_scenario == 'constant':
        first_load = np.concatenate((np.linspace(0, load * S_max1, t_steps_cycle), np.linspace(
            load * S_max1, load * S_min1, t_steps_cycle)[1:]))

        cycle1 = np.concatenate(
            (np.linspace(load * S_min1, load * S_max1, t_steps_cycle)[1:],
             np.linspace(load * S_max1, load * S_min1, t_steps_cycle)[
             1:]))
        cycle1 = np.tile(cycle1, n_cycles1 - 1)

        sin_load = np.concatenate((first_load, cycle1))

    if loading_scenario == 'order':
        first_load = np.concatenate((np.linspace(0, load * S_max1, t_steps_cycle), np.linspace(
            load * S_max1, load * S_min1, t_steps_cycle)[1:]))

        cycle1 = np.concatenate(
            (np.linspace(load * S_min1, load * S_max1, t_steps_cycle)[1:],
             np.linspace(load * S_max1, load * S_min1, t_steps_cycle)[
             1:]))
        cycle1 = np.tile(cycle1, np.int(np.floor(eta1 * cycles1)) - 1)

        change_order = np.concatenate(
            (np.linspace(load * S_min1, load * S_max2, 632)[1:], np.linspace(load * S_max2, load * S_min1, 632)[
                                                                 1:]))

        cycle2 = np.concatenate(
            (np.linspace(load * S_min1, load * S_max2, t_steps_cycle)[1:],
             np.linspace(load * S_max2, load * S_min1, t_steps_cycle)[
             1:]))
        cycle2 = np.tile(cycle2, n_cycles2)

        sin_load = np.concatenate((first_load, cycle1, change_order, cycle2))

    t_steps = len(sin_load)
    T = 1 / n_cycles1
    # t = np.linspace(0, 1, len(sin_load))

    start = time.time()

    U, F, cyc, number_cyc = get_UF_t(
        sin_load, t_steps, load, S_max1, S_max2, S_min1, tmodel, loading_scenario)

    end = time.time()
    print(end - start, 'seconds')

    # [omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z_N_Emn, X_N_Emn, Y_N_Emn, omega_T_Emn, z_T_Emn,
    #  alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Z_T_pi_Emn, X_T_pi_Emna, Y_T_pi_Emn, Disip_omena_N_Emn, Disip_omena_T_Emn,
    #  Disip_eps_p_N_Emn, Disip_eps_p_T_Emn, Disip_iso_N_Emn, Disip_iso_T_Emn, Disip_kin_N_Emn, Disip_kin_T_Emn] \
    #     = get_int_var(path, len(F), n_mp)

    font = {'family': 'DejaVu Sans',
            'size': 18}

    matplotlib.rc('font', **font)

    print(np.max(np.abs(F[:, 0])), 'sigma1')
    print(np.max(np.abs(F[:, 1])), 'sigma2')
    print(cyc * n_cycles1, 'cycles')

    # Fig 1, stress-strain curve

    # plt.plot(np.arange(len(F[:, 0])),np.abs(F[:, 0])-np.abs(F_int[:, 0]))
    # plt.show()

    f, (ax2) = plt.subplots(1, 1, figsize=(5, 4))

    ax2.plot(np.abs(U[:, 0]), np.abs(F[:, 0]), linewidth=2.5)
    ax2.set_xlabel(r'$|\varepsilon_{11}$|', fontsize=25)
    ax2.set_ylabel(r'$|\sigma_{11}$| [-]', fontsize=25)
    ax2.set_title('stress-strain' + ',' + 'N =' + str(cyc * n_cycles1) + str(S_max1) + 'Smin=' + str(S_min1))
    plt.show()

    if loading_scenario == 'constant':
        # Fig 2, creep fatigue curve

        f, (ax) = plt.subplots(1, 1, figsize=(5, 4))

        ax.plot((np.arange(len(U[2::2, 0])) + 1) / len(U[2::2, 0]),
                np.abs(U[2::2, 0]), linewidth=2.5)

        ax.set_xlabel(r'$N / N_f $|', fontsize=25)
        ax.set_ylabel(r'$|\varepsilon_{11}^{max}$|', fontsize=25)
        ax.set_title('creep fatigue Smax=' + str(S_max1) + 'Smin=' + str(S_min1))

        plt.show()

    if loading_scenario == 'order':
        # Fig 2, Creep Fatigue Curve

        f, (ax) = plt.subplots(1, 1, figsize=(5, 4))

        X_axis1 = np.array(np.arange(eta1 * cycles1) + 1)[:] / cycles1
        # X_axis1 = np.concatenate((np.array([0]), X_axis1))
        Y_axis1 = np.abs(U[3:np.int(2 * eta1 * cycles1) + 2:2, 0])
        # Y_axis1 = np.concatenate((np.array([Y_axis1[0]]), Y_axis1))

        X_axis2 = np.array((np.arange(len(U[2::2, 0]) - (eta1 * cycles1)) + 1) / (cycles2) + eta1)
        X_axis2 = np.concatenate((np.array([X_axis1[-1]]), X_axis2))
        Y_axis2 = np.abs(U[np.int(2 * eta1 * cycles1) + 1::2, 0])
        Y_axis2 = np.concatenate((np.array([Y_axis2[0]]), Y_axis2))

        ax.plot(X_axis1, Y_axis1, 'k', linewidth=2.5)
        ax.plot(X_axis2, Y_axis2, 'k', linewidth=2.5)
        ax.plot([X_axis1[-1], X_axis2[0]], [Y_axis1[-1], Y_axis2[0]], 'k', linewidth=2.5)

        ax.set_ylim(0.002, 0.0045)
        ax.set_xlim(-0.1, 1.1)
        ax.set_xlabel('N/Nf', fontsize=25)
        ax.set_ylabel('strain', fontsize=25)
        plt.title('creep fatigue Smax1=' + str(S_max1) + 'Smax2=' + str(S_max2) + 'Smin=' + str(S_min1))
        plt.show()

        plt.show()

