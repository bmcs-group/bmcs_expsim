import time
from bmcs_expsim.utils.mlab_decorators import decorate_figure
from mayavi import mlab
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import warnings
import matplotlib.pylab as plt
import numpy as np
from ibvpy.api import TStepBC , TFCyclicNonsymmetricConstant, TFBilinear
from ibvpy.bcond import BCSlice, BCDof
from ibvpy.xmodel.xdomain_fe_grid_axisym import XDomainFEGridAxiSym
from ibvpy.xmodel.xdomain_fe_grid import XDomainFEGrid
from ibvpy.xmodel.xdomain_interface import XDomainFEInterface
from ibvpy.fets import FETS2D4Q
from ibvpy.fets.fets1D5 import FETS1D52ULRH
from ibvpy.tmodel.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from ibvpy.tmodel.viz3d_tensor_field import \
    Vis3DTensorField, Viz3DTensorField
from bmcs_matmod.slide.vslide_34_TN_axisym import Slide34
from ibvpy.tmodel.mats3D.mats3D_elastic.vmats3D_elastic import \
    MATS3DElastic
from ibvpy.tmodel.mats2D.mats2D_elastic.vmats2D_elastic import \
    MATS2DElastic


n_y_e = 4
n_inner_x_e = 4
n_outer_x_e = 4
L_x = 40.0
R_in = 25.0
R_out = 50.0
P = 1.

xd_lower = XDomainFEGrid(integ_factor = P,
                         coord_min=(0, 0),
                          coord_max=(L_x, R_in),
                          shape=(n_y_e, n_inner_x_e),
                          fets=FETS2D4Q())
xd_upper = XDomainFEGrid(integ_factor = P,
                         coord_min=(0, R_in),
                          coord_max=(L_x, R_out),
                          shape=(n_y_e, n_outer_x_e),
                          fets=FETS2D4Q())
m1 = MATS2DElastic(E=37000, nu=0.18)
m2 = MATS2DElastic(E=37000, nu=0.18)

xd12 = XDomainFEInterface(
    integ_factor = P,
    I=xd_lower.mesh.I[1:-1, -1],
    J=xd_upper.mesh.I[1:-1, 0],
    fets=FETS1D52ULRH()
)
# material_params =  dict(
#      E_T=1000, gamma_T=100000, K_T=0, S_T=0.2, c_T=2, bartau=5,
#      E_N=1000, S_N=0.05, c_N = 9, m = 0.3, f_t=2, f_c=100, f_c0 = 80, eta=0.0)

material_params =  dict(
     E_T=800, gamma_T=6000, K_T=0, S_T=0.2, c_T=5, bartau=8,
     E_N=800, S_N=0.05, c_N = 9, m = 0.55, f_t=2, f_c=100, f_c0 = 80, eta=0.0)



bond_m = Slide34(**material_params)

m = TStepBC(
        domains=[(xd_lower, m1),
                 (xd_upper, m2),
                 (xd12, bond_m),
                 ]
    )


lower_fixed_0 = BCSlice(slice=xd_lower.mesh[:, 0, :, 0], var='u', dims=[1], value=0)
upper_fixed_1 = BCSlice(slice=xd_upper.mesh[0, :, 0, :], var='u', dims=[0], value=0)

tf_first = TFBilinear(loading_ratio=1.0, time_ratio=0.05)
tf_second = TFBilinear(loading_ratio=0.00, time_ratio=0.05)

list_param = np.linspace(300,500,3)
compression_list = np.linspace(-15,-30,2)
list_param = [500]


for param in list_param:

    material_params['gamma_T'] = param
    bond_m = Slide34(**material_params)

    m_list_mon = []
    max_slide = 2.0

    for c in compression_list:
        m = TStepBC(
            domains=[(xd_lower, m1),
                     (xd_upper, m2),
                     (xd12, bond_m),
                     ]
        )

        lower_slide = BCSlice(slice=xd_lower.mesh[0, :, 0, :], var='u', dims=[0], value=max_slide, time_function=tf_second)
        push = 1500 / len(lower_slide.dofs)
        push_force = [BCDof(var='f', dof=dof, value=push, time_function=tf_second)
                      for dof in lower_slide.dofs]

        upper_compression_slice = BCSlice(slice=xd_upper.mesh[:, -1, :, -1],
                 var='u', dims=[1], value=-2/material_params['f_c'],  time_function=tf_first)

        compression_dofs = upper_compression_slice.dofs

        compression = c * 20 / len(compression_dofs)
        upper_compression_force_first = [BCDof(var='f', dof=dof, value = compression, time_function=tf_first)
                     for dof in compression_dofs ]

        upper_compression_force_first[0].value = upper_compression_force_first[0].value/2
        upper_compression_force_first[-1].value = upper_compression_force_first[-1].value/2

        bc1 = [lower_fixed_0, upper_fixed_1] + upper_compression_force_first + push_force

        m.bc=bc1

        s = m.sim
        s.tloop.verbose = True
        s.tloop.k_max = 1000
        s.tline.step = 0.02
        s.tstep.fe_domain.serialized_subdomains

        xd12.hidden = True
        s.reset()
        try:
            s.run()
        except:
            pass
        m_list_mon.append(m)

max_tau = []


f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
#param = 0
for m,c in zip(m_list_mon,compression_list):
    F_to = m.hist.F_t
    U_to = m.hist.U_t
    F_inner_t = np.sum(F_to[:, lower_slide.dofs], axis=-1)
    U_inner_t = np.average(U_to[:, lower_slide.dofs], axis=-1)
    print(max(F_inner_t))
    max_tau.append(max(F_inner_t))

    states_t = [states_t[2] for states_t in m.hist.state_vars]
    var_names = states_t[0].keys()
    EpsSig_t = {
        var_name: np.array([state_dict[var_name] for state_dict in states_t])
        for var_name in var_names
    }
    u_pi_N = EpsSig_t['w_pi']
    u_pi_Tx = EpsSig_t['s_pi_x']
    sig_pi_N = EpsSig_t['sig_pi']
    sig_pi_Tx = EpsSig_t['tau_pi_x']
    sig_pi_Ty = EpsSig_t['tau_pi_y']
    omega_Nx = EpsSig_t['omega_N']
    omega_Tx = EpsSig_t['omega_T']
    x_m = xd12.x_Eia[:, :, 0].flatten()

    time = m.hist.t
    ax1.plot(time, sig_pi_N[:, 0, 0].flatten(), label='sigma'+ str(param)+ str(c))
    ax1.set_xlabel('time')
    ax1.set_ylabel('normal stress')
    ax1.legend()

    ax2.plot(U_inner_t, abs(sig_pi_Tx[:, 0, 0].flatten()), color='red', label='tau'+ str(param)+ str(c))
    ax2.set_xlabel('displacement [mm]')
    ax2.set_ylabel('tangential stress')
    ax2.legend()

    ax3.plot(time, omega_Nx[:, 0, 0].flatten(), color='red', label='normal damage'+ str(param)+ str(c))
    ax3.plot(time, omega_Tx[:, 0, 0].flatten(), color='green', label='tangential damage'+ str(param)+ str(c))

    ax3.set_xlabel('time')
    ax3.set_ylabel('damage')
    ax3.legend()

    ax4.plot(U_inner_t, F_inner_t * 2 * 25 * np.pi, color='red', label='F-U'+ str(param)+ str(c))
    ax4.set_xlabel('displacement [mm]')
    ax4.set_ylabel('applied force')
    ax4.legend()

plt.show()


n_y_e = 4
n_inner_x_e = 4
n_outer_x_e = 4
L_x = 40.0
R_in = 25.0
R_out = 50.0
P = 1.

xd_lower = XDomainFEGrid(integ_factor = P,
                         coord_min=(0, 0),
                          coord_max=(L_x, R_in),
                          shape=(n_y_e, n_inner_x_e),
                          fets=FETS2D4Q())
xd_upper = XDomainFEGrid(integ_factor = P,
                         coord_min=(0, R_in),
                          coord_max=(L_x, R_out),
                          shape=(n_y_e, n_outer_x_e),
                          fets=FETS2D4Q())
m1 = MATS2DElastic(E=37000, nu=0.18)
m2 = MATS2DElastic(E=37000, nu=0.18)

xd12 = XDomainFEInterface(
    integ_factor = P,
    I=xd_lower.mesh.I[1:-1, -1],
    J=xd_upper.mesh.I[1:-1, 0],
    fets=FETS1D52ULRH()
)


bond_m = Slide34(**material_params)

m = TStepBC(
        domains=[(xd_lower, m1),
                 (xd_upper, m2),
                 (xd12, bond_m),
                 ]
    )


lower_fixed_0 = BCSlice(slice=xd_lower.mesh[:, 0, :, 0], var='u', dims=[1], value=0)
upper_fixed_1 = BCSlice(slice=xd_upper.mesh[0, :, 0, :], var='u', dims=[0], value=0)

cycles = 10000
tf_precrompression = TFCyclicNonsymmetricConstant(number_of_cycles = cycles, shift_cycles = 0, unloading_ratio = 1.0)
tf_cyclic = TFCyclicNonsymmetricConstant(number_of_cycles = cycles, shift_cycles = 1, unloading_ratio = 0.2)


list_param = np.linspace(300,500,3)
compression_list = np.linspace(-15,-30,2)
#list_param = [500]

bond_m = Slide34(**material_params)

m_list = []
# max_tau = [1191.7096657386546, 1308.0563250091159]
S_max = 0.85

for (c,p) in zip(compression_list,max_tau):
    m = TStepBC(
        domains=[(xd_lower, m1),
                 (xd_upper, m2),
                 (xd12, bond_m),
                 ]
    )

    lower_slide = BCSlice(slice=xd_lower.mesh[0, :, 0, :], var='u', dims=[0], value=.8, time_function=tf_cyclic)

    push = p * S_max / len(lower_slide.dofs)
    push_force = [BCDof(var='f', dof=dof, value=push, time_function=tf_cyclic)
                                     for dof in lower_slide.dofs]

    upper_compression_slice = BCSlice(slice=xd_upper.mesh[:, -1, :, -1],
             var='u', dims=[1], value=-2/material_params['f_c'],  time_function=tf_precrompression)

    compression_dofs = upper_compression_slice.dofs

    compression = c * 20 / len(compression_dofs)
    upper_compression_force_first = [BCDof(var='f', dof=dof, value = compression, time_function=tf_precrompression)
                 for dof in compression_dofs ]

    upper_compression_force_first[0].value = upper_compression_force_first[0].value/2
    upper_compression_force_first[-1].value = upper_compression_force_first[-1].value/2

    bc1 = [lower_fixed_0, upper_fixed_1] + upper_compression_force_first + push_force

    m.bc=bc1
    m.hist.vis_record = {
    #    'strain': Vis3DTensorField(var='eps_ab'),
        'stress': Vis3DTensorField(var='sig_ab'),
        #        'kinematic hardening': Vis3DStateField(var='z_a')
    }

    s = m.sim
    s.tloop.verbose = True
    s.tloop.k_max = 1000
    s.tline.step = 1/(50*2*cycles)
    s.tstep.fe_domain.serialized_subdomains

    xd12.hidden = True
    s.reset()
    try:
        s.run()
    except:
        pass
    m_list.append(m)
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

for m,c in zip(m_list,compression_list):
    F_to = m.hist.F_t
    U_to = m.hist.U_t
    F_inner_t = np.sum(F_to[:, lower_slide.dofs], axis=-1)
    U_inner_t = np.average(U_to[:, lower_slide.dofs], axis=-1)
    print(max(F_inner_t))

    states_t = [states_t[2] for states_t in m.hist.state_vars]
    var_names = states_t[0].keys()
    EpsSig_t = {
        var_name: np.array([state_dict[var_name] for state_dict in states_t])
        for var_name in var_names
    }
    u_pi_N = EpsSig_t['w_pi']
    u_pi_Tx = EpsSig_t['s_pi_x']
    sig_pi_N = EpsSig_t['sig_pi']
    sig_pi_Tx = EpsSig_t['tau_pi_x']
    sig_pi_Ty = EpsSig_t['tau_pi_y']
    omega_Nx = EpsSig_t['omega_N']
    omega_Tx = EpsSig_t['omega_T']
    x_m = xd12.x_Eia[:, :, 0].flatten()

    time = m.hist.t
    ax1.plot(U_inner_t, sig_pi_N[:, 0, 0].flatten(), label='sigma'+  str(c))
    ax1.set_xlabel('dispacement [mm]')
    ax1.set_ylabel('normal stress')
    ax1.legend()

    ax2.plot(U_inner_t, sig_pi_Tx[:, 0, 0].flatten(), label='tau'+  str(c))
    ax2.set_xlabel('dispacement [mm]')
    ax2.set_ylabel('tangential stress')
    ax2.legend()

    ax3.plot(time, omega_Nx[:, 0, 0].flatten(), label='normal damage'+ str(c))
    ax3.plot(time, omega_Tx[:, 0, 0].flatten(), label='tangential damage'+ str(c))

    ax3.set_xlabel('time')
    ax3.set_ylabel('damage')
    ax3.legend()

    ax4.plot(U_inner_t, F_inner_t , label='F-U'  + str(c))
    ax4.set_xlabel('dispacement [mm]')
    ax4.set_ylabel('applied force')
    ax4.legend()

plt.show()

