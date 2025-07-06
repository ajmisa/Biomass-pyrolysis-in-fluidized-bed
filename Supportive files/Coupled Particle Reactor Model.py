# Import necessary libraries
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import scipy as sp
import scipy.sparse.linalg as sla
from scipy.optimize import fsolve
from pymrm import construct_grad, construct_convflux_upwind, interp_cntr_to_stagg_tvd, minmod, construct_div, newton, construct_coefficient_matrix, NumJac, vanleer, construct_convflux_upwind
import numpy as np
from pymrm.solve import newton


class ParticleModel:
    """
    1D diffusion-convection-reaction model with nonlinear kinetics, dynamic porosity and heat transfer.

    This class solves transient or steady-state mass and energy balances in a porous spherical particle.
    It supports arbitrary nonlinear reaction kinetics, time-dependent porosity, and temperature-dependent
    effective conductivity (conduction + radiation). 

    It builds the full Jacobian matrix and residual vector for implicit time integration with Newton's method.
    The model can be used for single particles or coupled to reactor-scale simulations.

    Parameters
    ----------
    shape : tuple
        Shape of the concentration field (number of grid points, number of components).
    axis : int, optional
        Axis along which diffusion and convection occur (default: 0).
    L : float, optional
        Length of the domain (default: 1.0).
    v : float or array, optional
        Velocity profile (default: 1.0).
    D : float or array, optional
        Diffusion coefficients (default: 0.0).
    c_0 : float or array, optional
        Initial concentration field (default: 0.0).
    h_mass : float, optional
        Mass transfer coefficient for outer boundary (default: 1.0).
    T_g : float, optional
        External gas temperature for boundary condition (default: 873.0).
    rho_bulk : array, optional
        Bulk gas density for boundary mass transfer (default: None).
    dt : float, optional
        Time step size (default: np.inf for steady-state).
    callback_newton : callable, optional
        Callback for Newton iteration (default: None).

    Attributes
    ----------
    c : ndarray
        Concentration field [kg/m³] and Temperature [K] for all components.
    c_old : ndarray
        Previous concentration field (for time stepping).
    z_f, z_c : ndarray
        Face and center grid points.
    eps_p : ndarray
        Local porosity field.
    jac_const : sparse matrix
        Constant part of the Jacobian.
    g_const : ndarray
        Constant part of the residual.
    """

    def __init__(self, shape, axis=0, L=1.0, v=1.0, D=0.0,
                 c_0=0.0, T_g=873.0, Re= 1.0, rho_bulk=None, dt=np.inf, callback_newton=None):
        # Model parameters
        self.shape = shape
        self.axis = axis
        self.L = L
        self.v = np.asarray(v)
        self.D = np.asarray(D)
        self.dt = dt
        self.T_g = T_g
        self.rho_bulk = np.asarray(rho_bulk)
        self.Re = Re

        # Porosity model constants
        self.eps_B0 = 0.4
        self.eps_Cf = 0.91
        self.rho_B0 = c_0[0,0]
        self.rho_Cf = c_0[0,0]
        self.kg = 2.577e-2
        self.viscos_bulk = 123.37e-6 # m^2/s
        
        # Grid initialization
        self.z_f = np.linspace(0, self.L, self.shape[self.axis] + 1)
        self.z_c = 0.5 * (self.z_f[:-1] + self.z_f[1:])
        self.init_field(c_0)
    
        # Jacobian helper
        self.callback_newton = callback_newton
        self.numjac = NumJac(shape)
    
    def init_field(self, c_0=0):
        """
        Initialize the concentration field with a uniform value.
        """
        # Concentration field
        c = np.asarray(c_0)
        shape = (1,) * (len(self.shape) - c.ndim) + c.shape
        c = c.reshape(shape)
        self.c = np.broadcast_to(c, self.shape).copy()
        self.c_old = self.c.copy()
    
        # Porosity field
        self.eps_p = np.ones(self.shape)
        self.eps_p_old = np.ones(self.shape)
        for j in [1, 2, 5, 6]: # set porosity for all gas species
            self.eps_p[:, j] = self.eps_B0
            self.eps_p_old[:, j] = self.eps_B0

    def update_kinetics(self, c, eps_p, Cp_total):
        """
        Compute reaction rates and heat source terms, returns R_out [kg/m³/s]
        """
        c = c.reshape(self.shape)
        B, Tar, G, C, M, W, I, T = [c[..., j] for j in range(8)]
    
        # Kinetics parameters
        p = {
            'A1': 4.38e9, 'E1': 152.7e3,
            'A2': 1.08e10, 'E2': 147e3,
            'A3': 3.27e6, 'E3': 111.7e3,
            'A4': 4.28e6, 'E4': 107.5e3,
            'A5': 1e5, 'E5': 108e3,
            'Avap': 5.13e10, 'Evap': 88e3,
            'R': 8.314
        }
    
        R = p['R']
        k = {
            'K1': p['A1'] * np.exp(-p['E1'] / (R * T)),
            'K2': p['A2'] * np.exp(-p['E2'] / (R * T)),
            'K3': p['A3'] * np.exp(-p['E3'] / (R * T)),
            'K4': p['A4'] * np.exp(-p['E4'] / (R * T)),
            'K5': p['A5'] * np.exp(-p['E5'] / (R * T)),
            'Kvap': p['Avap'] * np.exp(-p['Evap'] / (R * T)),
        }
    
        delta_H = {
            'dH1': 64000.0, 'dH2': 64000.0, 'dH3': 64000.0,
            'dH4': -42000.0, 'dH5': -42000.0, 'dHvap': 2.44e6
        }
    
        # Reaction rates
        R1 = k['K1'] * B
        R2 = k['K2'] * B
        R3 = k['K3'] * B
        R4 = k['K4'] * eps_p[..., 1] * Tar
        R5 = k['K5'] * eps_p[..., 1] * Tar
        Rvap = k['Kvap'] * M
    
        # Mass source terms
        R_out = np.zeros_like(c)
        R_out[..., 0] = -(R1 + R2 + R3)
        R_out[..., 1] = R2 - (R4 + R5)
        R_out[..., 2] = R1 + R4
        R_out[..., 3] = R3 + R5
        R_out[..., 4] = -Rvap
        R_out[..., 5] = Rvap
        R_out[..., 6] = 0
    
        # Energy source term
        Qdot = -(
            R1 * delta_H['dH1'] + R2 * delta_H['dH2'] + R3 * delta_H['dH3'] +
            R4 * delta_H['dH4'] + R5 * delta_H['dH5'] + Rvap * delta_H['dHvap']
        )
        R_out[..., 7] = Qdot / (Cp_total + 1e-8)
    
        return R_out

    def update_porosity(self):
        """
        Update porosity field based on biomass conversion for gas-phase components only.
        Solid-phase components (B, M, C) keep eps_p = 1.
        """
        rho_B = self.c[:, 0]
        rho_C = self.c[:, 3]
        XB = (self.rho_B0 - (rho_B + rho_C)) / (self.rho_B0 - self.rho_Cf + 1e-12)
    
        self.XB = np.clip(XB, 0.0, 1.0)
        eps_dynamic = self.eps_B0 * (1 - self.XB) + self.eps_Cf * self.XB
        eps_dynamic = np.clip(eps_dynamic, self.eps_B0, self.eps_Cf)
    
        self.eps_p = np.ones_like(self.c)
    
        # Gas phase components
        for j in [1, 2, 5, 6]:
            self.eps_p[:, j] = eps_dynamic

    def update_bc(self):
            """
            Update boundary conditions using current effective diffusivity D_eff = eps_p * D,
            and apply Robin BC for heat transfer (convection + radiation) at particle surface.
            Uses linearized radiation term for numerical stability.
            """
            Nc = self.shape[1]
            dpore_boundary = 5e-5 # m
        
            # Effective diffusion
            D_eff_center = self.eps_p * self.D
            D_eff_center[:, -1] = self.keff / self.Cp_total
            D_eff_face = 0.5 * (D_eff_center[:-1] + D_eff_center[1:])
            D_face_L = D_eff_center[0]
            self.D_face_R = 2 * D_eff_face[-1] - D_eff_face[-2]
            self.D_eff_face = np.vstack([D_face_L, D_eff_face, self.D_face_R])

            Nu = 0.03 * self.Re ** 1.3 # valid if Reynolds < 100 (expected Re 3.2)
            h_conv = Nu*self.kg / (dpore_boundary)

            Sc = (self.viscos_bulk) / ( self.D_face_R[6] + 1e-6)
            Sh = 2 + 0.6 * Sc ** (1/3) * self.Re**(1/2)

            self.h_mass = Sh * self.D_face_R[6] / (dpore_boundary)
 
            # Heat transfer
            sigma = 5.67e-8
            ems = 0.9
            T_surface = self.c[-1, 7]
        
            h_rad = 4 * ems * sigma * T_surface**3
            h_eff = h_conv + h_rad
        
            # Left BC (r = 0): zero flux
            bc_L = {
                'a': [[1 if i in [1, 2, 5, 6, 7] else 0 for i in range(Nc)]],
                'b': [[0] * Nc],
                'd': [[0.0] * Nc]
            }
        
            # Right BC (r = R): Robin BC for gas & temperature
            bc_R = {
                'a': [[
                    self.D_face_R[i] if i in [1, 2, 5, 6] else
                    -self.keff[-1] if i == 7 else 0
                    for i in range(Nc)]
                ],
                'b': [[
                    -self.h_mass if i in [1, 2, 5, 6] else 
                    -h_eff if i == 7 else 0
                    for i in range(Nc)]
                ],
                'd': [[
                    -self.h_mass * self.rho_bulk[i] if i in [1, 2, 5, 6] else
                    -h_eff * self.T_g if i == 7 else 0
                    for i in range(Nc)]
                ]
            }
        
            self.bc = (bc_L, bc_R)
        

    def update_heat_capacity(self):
        """
        Update total heat capacity term Cp_total [J/m³·K] for energy balance:
        """

        Cp_B = 2300.0
        Cp_C = 1700.0
        Cp_G = 1250.0
        Cp_T = 2500.0
        Cp_W = 1996.0
        Cp_I = 1040.0
        Cp_M = 4180.0
    
        rho_B = self.c[:, 0]
        rho_T = self.c[:, 1]
        rho_G = self.c[:, 2]
        rho_C = self.c[:, 3]
        rho_M = self.c[:, 4]
        rho_W = self.c[:, 5]
        rho_I = self.c[:, 6]
    
        # Gas phase
        self.rho_total_g = rho_G + rho_T + rho_W + rho_I
        self.Cp_g_mix = (
            rho_G * Cp_G +
            rho_T * Cp_T +
            rho_W * Cp_W +
            rho_I * Cp_I
        ) / (self.rho_total_g + 1e-6)
    
        # Total Cp
        self.Cp_total = (
            rho_B * Cp_B +
            rho_C * Cp_C +
            rho_M * Cp_M +
            self.eps_p[:, 2] * self.rho_total_g * self.Cp_g_mix
        )


    def update_vT_profile(self):
        """
        Update convection velocity term for Temperature equation:
        """
        # Compute v_T(r)
        self.vT_term = self.rho_total_g * self.Cp_g_mix / (self.Cp_total + 1e-8)

    def update_heatcoeff(self):
        T = self.c[:, -1]
    
        kB0_along = 0.255
        kB0_across = 0.1046
        kCf_along = 0.105
        kCf_across = 0.071
        wb0 = 0.6
        wcf = 1
        dpore_B0 = 5e-5 # m
        dpore_Cf = 1e-3 # m
        sigma = 5.67e-8
    
        self.dpore = dpore_B0 * (1 - self.XB) + dpore_Cf * self.XB
        wp = wb0 * (1 - self.XB) + wcf * self.XB
        kB0 = 0.5 * (kB0_along + kB0_across)
        kCf = 0.5 * (kCf_along + kCf_across)
    
        # Thermal conductivity
        kp = kB0 * (1 - self.XB) + kCf * self.XB
        kcond = kp + self.eps_p[:, 1] * self.kg
    
        # Radiation term
        krad = (4 * wp * self.eps_p[:, 1] * sigma * T**3 * self.dpore) / (1 - self.eps_p[:, 1] + 1e-8)
    
        self.keff = kcond + krad

    def update_jac(self):
        """
        Construct the Jacobian matrix and constant terms for the system.
        """
    
        vT_face = 0.5 * (self.vT_term[:-1] + self.vT_term[1:])
        vT_faces = np.tile(self.v, (self.shape[0] + 1, 1)) #(Nr + 1, Nc)
        vT_faces[1:-1, 7] = vT_face #T no edges
        vT_faces[0, 7] = self.vT_term[0] #1st edge
        vT_faces[-1, 7] = 2 * vT_face[-1] - vT_face[-2] #N+1 linear extrapolation
        self.vT_faces = vT_faces
        v_faces = self.v * vT_faces # (Nc) * (Nr + 1, Nc)

        # Construct the Jacobian matrix and constant terms for the system
        grad_mat, grad_bc  = construct_grad(self.shape, self.z_f, self.z_c, self.bc, axis=self.axis)  # Gradient operator
        self.div_mat = construct_div(self.c.shape, self.z_f,nu=2, axis=self.axis)  # Divergence operator
        diff_mat = construct_coefficient_matrix(self.D_eff_face, shape=self.shape, axis=self.axis)
        convflux_mat, convflux_bc = construct_convflux_upwind(self.shape, self.z_f, self.z_c, self.bc, axis=self.axis, v = v_faces)  # Convection flux operator 
        jac_convdiff = self.div_mat @ (convflux_mat -diff_mat @ grad_mat) #- div_v_mat # Diffusion term
        self.g_const = self.div_mat @ (convflux_bc -diff_mat @ grad_bc)  # Boundary condition forcing term
        jac_accum = construct_coefficient_matrix(1.0/self.dt, shape = self.shape)  # Accumulation term
        self.jac_const = jac_accum + jac_convdiff  # Total Jacobian matrix

    def g(self, c):
        """
        Compute the residual vector and Jacobian matrix for the current time step.
        """
        c_new = c.reshape(self.shape)
        self.c[...] = c_new
    
        # Update fields
        self.update_porosity()
        self.update_heat_capacity()
        self.update_vT_profile()
        self.update_kinetics(self.c, self.eps_p, self.Cp_total)
        self.update_heatcoeff()
        self.update_bc()
        self.update_jac()
    
        # TVD interpolation
        _ , dc_f = interp_cntr_to_stagg_tvd(self.c, self.z_f, self.z_c, self.bc, self.v, minmod)
        _ , deps_f = interp_cntr_to_stagg_tvd(self.eps_p, self.z_f, self.z_c, self.bc, self.v, minmod)

        # Convection term
        dg_conv = self.div_mat @ (deps_f * self.v * self.vT_faces * dc_f).reshape((-1, 1))
    
        # Assemble residual
        g = self.g_const + self.jac_const @ (self.eps_p * c).reshape(-1, 1) + dg_conv - (self.eps_p_old * self.c_old).reshape(-1, 1) / self.dt
    
        # Reaction source terms
        g_react, jac_react = self.numjac(lambda x: self.update_kinetics(x, self.eps_p, self.Cp_total), self.c)

        g -= g_react.reshape((-1, 1))
        jac = self.jac_const - jac_react
    
        return g, jac

    def step_dt(self):
        """
        Store the current concentration and porosity field as the previous values for time stepping.
        """
        self.c_old = self.c.copy()
        self.eps_p_old = self.eps_p.copy()

    def solve(self):
        """
        Solve the system using Newton's method.
        """
        result = newton(lambda c: self.g(c), self.c, maxfev=10, callback=self.callback_newton)
        self.c[...] = result.x.reshape(self.c.shape)  # Update the concentration field

    def app_rate_surface_conc(self):
        """
        Compute mass concentrations, diffusive molar‐flux,
        and volumetric molar‐flux at the outer (r=L) boundary.

        Returns
        -------
        c_surface : ndarray, shape (Nc,)
            Mass concentration [kg/m³] at the surface node.
        Rate     : ndarray, shape (Nc,)
            Apparent reaction rate [kg/m³/s]
        """

        self.update_heat_capacity()     # This initializes Cp_total
        Rate_space = self.update_kinetics(self.c, self.eps_p, self.Cp_total)
        Rate = (3 / self.z_c[-1]**3) * np.trapezoid(Rate_space[:, :] * self.z_c[:, np.newaxis]**2, self.z_c, axis=0)

        dr = self.z_f[1] - self.z_f[0]
        dc_dr = (self.c[-1, :] * self.eps_p[-1, :] - self.c[-2, :]*self.eps_p[-2, :]) / dr
        
        # Fluxes
        D_eff_surf = self.D_face_R

        J_mass = - D_eff_surf * dc_dr * 3 / self.L

        return Rate, J_mass

def main():
    # --- Physical Parameters ---
    L = 0.5         # Length of Reactor (m)
    Disp = 1e6           # Diffusion coefficient (m^2/s)
    t_end = 50
    R = 0.1             # Inner radius of the reactor (m)
    A_reactor = np.pi * R**2 # Cross sectionnal area of reactor (m^2)
    g = 9.81 #m^2/s acceleration due to gravity

    # parameters
    params = {
        'A4': 4.28e6, 
        'E4': 107.5e3,
        'A5': 1e5, 
        'E5': 108e3,
        'Avap': 5.13e10, 
        'Evap': 88e3,
        "D": 1e-6, #m^2/s
        "d_p": 500e-6, #particle size (m)
        "P": 101325, # Pa, atmospheric pressure
        "T": 600 + 273.15, # K, temperature (1000C)
        "R": 8.3145, # J/(mol*K), universal gas constant
        "mu": 4.6177456e-5,
        "rho_p": 570, # initial particle density
        "fw":  0.25, 
        "gamma_b": 0.005,
    }

    params["K4"] = params['A4'] * np.exp(-params['E4'] / (params["R"] * params["T"]))
    params["K5"] = params['A5'] * np.exp(-params['E5'] / (params["R"] * params["T"]))

    params["u0"] = 1.1 #superficial velocity (m/s)
    CN20 = params["P"]/(params["R"]*params["T"]) # 20C in K
    params["rho_N2"] = CN20 * 28.0134e-3 # kg/mol, molar mass of N2
    params["Re_p"] = (params["rho_N2"]*params["u0"]*params["d_p"])/params["mu"] # Reynolds number

    params["Ar"]=g*params["d_p"]**3*params["rho_N2"]*(params["rho_p"]-params["rho_N2"])/(params["mu"]**2)#Archimedes Number
    params["umf"] = (params["mu"]/(params["rho_N2"]*params["d_p"]))*(np.sqrt(33.7**2+0.0408*params["Ar"])-33.7)#mimumum fluidization velcoity: m/s

    def Ergun(ep_mf):
        a= (params["rho_p"]-params["rho_N2"])*g
        b=150*params["mu"]*params["umf"]/(params["d_p"]**2)
        c = 1.75*params["rho_N2"]*params["umf"]**2/params["d_p"]
        return -a*ep_mf**3-b*ep_mf+b+c
    
    params["ep_mf"] = fsolve(Ergun, 0.4)

    params["d_b"] = 1e-2 * 0.610*(1 + 1e2*0.272*(params["u0"] - params["umf"]))**(1/3)*(1 + 0.0684*L/2)**1.21 #bubble diameter (m)
    params["ubr"] = 0.711*np.sqrt(g*params["d_b"])
    params["ub"] = params["u0"]-params["umf"]+params["ubr"]#
    params["delta"] = (params["u0"]-params["umf"])/(params["ub"]-params["umf"]) 
    params["ue"] = params["umf"]/params["ep_mf"]
    params["gamma_c"] = (1 - params["ep_mf"])*(3/(params["ubr"]*params["ep_mf"]/params["umf"] - 1) + params["fw"]) 
    params["gamme_e"] = (1-params["ep_mf"])*(1 - params["delta"])/params["delta"] - params["gamma_b"] - params["gamma_c"]
    params["ep_s"] = (params["gamma_b"] + params["gamma_c"] + params["gamme_e"])*params["delta"]
    params["ep_e"] = 1 - params["delta"] - params["ep_s"] 
    params["bm_in"] = params["rho_p"]*params["ue"][0]*A_reactor
    params["water_in"] = 0.09*params["rho_p"]*params["ue"][0]*A_reactor

    params["Kbc"] = 4.5*(params["umf"]/params["d_b"])+5.85*(params["D"]**0.5*g**0.25/(params["d_b"]**1.25))
    params["Kce"] = 6.77*np.sqrt(params["D"]*params["ep_mf"]*params["ubr"]/(params["d_b"]**3))
    params["Kbe"] = 1/(1/params["Kbc"] +1/params["Kce"])

    params["Sc"] = params["mu"]/(params["rho_N2"]*params["D"])
    params["Sh"] = 2 + 0.552*params["Re_p"]**0.5*params["Sc"]**(1/3)

    params["Kep"] = params["Sh"]*params["D"]/params["d_p"]*6*params["ep_s"]/params["d_p"] #mass transfer coefficient for emulsion to particles

    # --- Discretization Parameters ---
    num_dt = 500
    dt_reactor = t_end / num_dt
    num_z = 50    # Number of axial grid points
    num_r = 30   # Number of radial grid points
    num_p = 3       #Numer of phases
    num_c = 6       # Number of species

    # --- Grid Generation ---
    z_f = np.linspace(0, L, num_z + 1)                    # Axial face grid
    z_c = 0.5 * (z_f[1:] + z_f[:-1])                      # Axial cell-centered grid
    r_f = np.linspace(0, R, num_r + 1)                # Radial face grid
    r_c = 0.5 * (r_f[1:] + r_f[:-1])          # Radial cell-centered grid

    # --- Boundary Conditions ---
    bc_neumann = {'a': 1, 'b': 0, 'd': 0}                  # No-flux boundary
    a = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [Disp, Disp, Disp, Disp, Disp, Disp]]
    b = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [params["ue"][0]/params["ep_s"][0], params["ue"][0]/params["ep_s"][0], params["ue"][0]/params["ep_s"][0], params["ue"][0]/params["ep_s"][0], params["ue"][0]/params["ep_s"][0], params["ue"][0]/params["ep_s"][0]]]  # Dirichlet boundary
    d = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [params["ue"][0]*params["rho_p"]/params["ep_s"][0], 0, 0, 0, params["ue"][0]*params["rho_p"]/params["ep_s"][0] *0.09, 0]]
    bc_inlet = {'a': a, 'b': b, 'd': d}  # Dirichlet boundary for inlet emulsion and bubble, and Dankwertz for solid phase
    
    # --- Matrix Shape and Numerical Jacobian Helper ---
    shape = (num_z, num_r, num_p, num_c)
    shape_nj = (num_z, num_r, num_p * num_c)
    numjac = NumJac(shape_nj)

    # --- Differential Operators ---
    div_z = construct_div(shape, z_f, axis=0, nu=0)      # Axial divergence
    div_r = construct_div(shape, r_f, axis=1, nu=1)  # Radial divergence

    # --- Convection Term ---
    Ub_prof = 2 * params["ub"] * (1 - (r_c / R) ** 2)  # Parabolic velocity profile
    Ue_prof = 2 * params["ue"] * (1 - (r_c / R) ** 2)  # Parabolic velocity profile

    UMAT = np.zeros((shape))
    for i in range(num_z):
        for j in range(num_c):
            UMAT[i,:,0,j] = Ub_prof
            UMAT[i,:,1,j] = Ue_prof
            UMAT[i,:,2,j] = params["ue"]
    conv, conv_bc = construct_convflux_upwind(shape, z_f, z_c, (bc_inlet, bc_neumann), UMAT, axis=0) 

    # interpolate UMAT from cell-centers to faces along axis=0
    UMAT_f = np.zeros((num_z + 1, num_r, num_p, num_c))
    UMAT_f[1:-1] = 0.5 * (UMAT[:-1] + UMAT[1:])
    UMAT_f[0] = UMAT[0]     # or appropriate boundary extrapolation
    UMAT_f[-1] = UMAT[-1]   # same as above
    
    D_mat = construct_coefficient_matrix([[0,0,0,0, 0, 0], [0,0,0,0, 0, 0], [Disp,Disp,Disp,Disp,Disp,Disp]], shape=shape, axis=1)
    D_mat_z = construct_coefficient_matrix([[0,0,0,0, 0, 0], [0,0,0,0, 0, 0], [Disp,Disp,Disp,Disp,Disp,Disp]], shape=shape, axis=0)

    # --- Allocate Concentration Field ---
    c = np.zeros(shape)

    # --- Diffusion Gradient ---
    grad, grad_bc = construct_grad(shape, r_f, r_c, bc=(bc_neumann, bc_neumann), axis=1)
    grad_z, grad_z_bc = construct_grad(shape, z_f, z_c, bc=(bc_inlet, bc_neumann), axis=0)
    # --- Build Linear System ---
    jac = (div_z @ conv - div_r @ (D_mat @ grad) - div_z @ (D_mat_z @ grad_z))
    _, jac_source = numjac(lambda c: source(c, params), c)
    jac -= jac_source
    jac += sp.sparse.eye(c.size, format='csc')/dt_reactor
    rhs_const = (div_z @ conv_bc - div_r @ (D_mat @ grad_bc) - div_z @ (D_mat_z @ grad_z_bc)).toarray() 

    # --- Configuration Particle Model ---
    Nr = 40        # Radial grid points
    Nc = 8          # Number of species
    Nt = 50
    t_end_pm = L/params["ue"][0]    # Total time [s]
    dt = t_end_pm/(Nt)       # Time step [s]
    Radius = params["d_p"] / 2  # Particle radius [m]
    R_univ = 8.314      # J/(mol·K)
    P_atm  = 101325.0   # Pa
    MW_N2  = 28e-3
    shape_pm = (Nr, Nc)
    T_g = params["T"]  # Gas temperature [K]

    D_vec = np.zeros(Nc)
    D_vec[1] = 1e-6
    D_vec[2] = 1e-6
    D_vec[5] = 1e-6
    D_vec[6] = 1e-6

    v_profile = np.zeros(Nc)
    v_profile[1] = 1e-4   # velocity for Gas component [m/s]
    v_profile[2] = 1e-4   # velocity for Gas component [m/s]
    v_profile[5] = 1e-4   # velocity for Gas component [m/s]
    v_profile[6] = 1e-4   # velocity for Gas component [m/s]
    v_profile[7] = 1e-4   # velocity for Gas component [m/s]

    # Initial conditions [kg/m³]
    c0 = np.zeros(shape_pm)
    c0[:, 0] = 570.0  # Biomass
    c0[:, 4] = c0[:, 0] * 0.09    # Moisture
    c0[:, 7] = 298    # Temp [K] 

    rho_bulk = np.zeros(Nc)  
    rho_bulk[6] = P_atm * MW_N2 / (R_univ * T_g)

    model = ParticleModel(
        shape=shape_pm,
        axis=0,
        L=Radius,
        v=v_profile,
        D=D_vec,
        c_0=c0,
        dt=dt,
        rho_bulk=rho_bulk,
        T_g = T_g,
        Re = params["Re_p"]
        )
    
    # --- Storage ---
    t_vals = np.linspace(dt, t_end_pm, Nt)
    all_conc = np.zeros((Nt, Nc, Nr))
    R_space = np.zeros((Nt, Nc))
    flux = np.zeros((Nt, Nc))

    # Weighted average: account for spherical volume element 4πr²dr
    # --- Time stepping ---
    for step in range(Nt):
        model.step_dt()
        model.solve()
        all_conc[step] = (model.c * model.eps_p).T
        R_space[step, :], flux[step, :]  = model.app_rate_surface_conc()
    R_apparent = np.trapz(R_space, t_vals, axis=0) / (t_vals[-1] - t_vals[0])
    flux_app = np.trapz(flux, t_vals, axis=0) / (t_vals[-1] - t_vals[0])
    print(np.max(flux_app))
    print(flux_app, R_apparent)
    external_source = particle_coupling(flux_app, R_apparent, shape, params)

    # --- Visualization Setup ---
    labels = [r'Biomass', r'Tar', r'Gas', r'Char', r'Water', r'Water Vapor']  
    phases = ['bubble', 'emulsion', 'particles']  # update if num_p = 3
    colors_plot = ['black', 'blue', 'orange', 'green', 'red', 'purple']
    styles = ['-', '--', '-.']

    # --- Setup ---
    fig, ax = plt.subplots(num_p, num_c, figsize=(5 * num_c, 4 * num_p), sharey=True, sharex=True)
    vmin, vmax = 0, 10
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = 'viridis'

    img = []
    for p in range(num_p):
        row = []
        for i in range(num_c):
            ax[p, i].set_title(f'{labels[i]} - {phases[p]}')
            ax[p, i].set_xlabel('r [m]')
            ax[p, i].set_ylabel('z [m]')
            data = c[:, :, p, i]
            im = ax[p, i].pcolormesh(r_f, z_f, data, shading='auto', cmap=cmap, norm=norm)
            row.append(im)
        img.append(row)

    fig.subplots_adjust(right=0.88)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Concentration [kg/m³]')
    plt.ion()
    plt.show()
    
    mass_flow = np.zeros((num_c, num_dt))
    # --- Time Loop with Live Plot ---
    output_interval = 2
    for step in range(num_dt):
        c_old = c.copy().reshape(-1, 1, order='C')
        _, dc_f = interp_cntr_to_stagg_tvd(c, z_f, z_c, (bc_inlet, bc_neumann), UMAT_f, vanleer, axis=0) 
        conv_deferred = div_z @ (UMAT_f * dc_f).reshape((-1, 1))
        jac_lu = sla.splu(jac)

        rhs = rhs_const + jac @ c.reshape(-1, 1, order='C') - c_old / dt_reactor + conv_deferred
        rhs -= external_source.reshape(-1, 1)    # Add external source term
        c -= jac_lu.solve(rhs).reshape(c.shape, order='C')
        if np.min(c) < 0:
            print(f"Negative concentration encountered at step {step}. Stopping simulation.")
            break   
        for p in range(num_p):
            for i in range(num_c):
                if p == 0:
                    mass_flow[i,step] += np.trapz(Ub_prof*c[-1, :, p, i]*np.pi*(r_f[1:]**2 - r_f[:-1]**2))  # Mass flow rate at the outlet (z = L) for each species
                if p == 1:
                    mass_flow[i,step] += np.trapz(Ue_prof*c[-1, :, p, i]*np.pi*(r_f[1:]**2 - r_f[:-1]**2))  # Mass flow rate at the outlet (z = L) for each species
                if p == 2:
                    mass_flow[i,step] += np.trapz(params["ue"]*c[-1, :, p, i]*np.pi*(r_f[1:]**2 - r_f[:-1]**2))  # Mass flow rate at the outlet (z = L) for each species
            c
        if step % output_interval == 0 or step == num_dt - 1:
            for p in range(num_p):
                for i in range(num_c):
                    data = c[:, :, p, i]  # No padding
                    img[p][i].set_array(data.ravel())  # Just update the data
                    img[p][i].set_clim(vmin=vmin, vmax=vmax)
            fig.suptitle(f"Time: {step * dt_reactor:.2f} s")
            plt.pause(0.01)

    plt.ioff()
    plt.show()
    labels = [r'Biomass', r'Tar', r'Gas', r'Char', r'Water', r'Water Vapor']

    t = np.linspace(0, t_end, num_dt)
    # --- Plot Outlet Concentrations at z = z_end for both phases
    for i in range(1,num_c - 2):
        plt.plot(t,mass_flow[i, :]/params["bm_in"], label=labels[i])

    plt.xlabel('t [s]')
    plt.ylabel('yield per kg Dry Biomass [kg/kg]')
    plt.legend()
    plt.tight_layout()
    plt.show()

def particle_coupling(flux, R_apparent, shape, params):
    external_source = np.zeros(shape)

    external_source[:,:,1,1] = flux[1]*params["ep_e"]  # Tar in particles
    external_source[:,:,1,2] = flux[2]*params["ep_e"]   # Gas in particles
    external_source[:,:,1,5] = flux[5]*params["ep_e"]   # Gas in particles

    external_source[:,:,2,0] = R_apparent[0]
    external_source[:,:,2,1] = R_apparent[1] - flux[1]*params["ep_s"]  # Tar in emulsion
    external_source[:,:,2,2] = R_apparent[2] - flux[2]*params["ep_s"]
    external_source[:,:,2,3] = R_apparent[3]
    external_source[:,:,2,4] = R_apparent[4]
    external_source[:,:,2,5] = R_apparent[5] - flux[5]*params["ep_s"]

    return external_source

def source(c, params):
    """
    Computes the rate of change for a sequential reaction A → B → C.

    Parameters:
        c     : ndarray - concentration array with shape (..., 3, 2)
        kf_1  : float   - forward rate constant for A → B
        kf_2  : float   - forward rate constant for B → C
        axis  : int     - axis along which species are stored

    Returns:
        s     : ndarray - rate of change for [A, B, C]
    """

    s = np.zeros_like(c)
    BM_in_Bub = c[:, :, 0, 0]
    BM_in_Emul = c[:, :, 1, 0]
    BM_in_Part = c[:, :, 2, 0]
    TAR_in_Bub = c[:, :, 0, 1]
    TAR_in_Emul = c[:, :, 1, 1]
    TAR_in_Part = c[:, :, 2, 1]
    GAS_in_Bub = c[:, :, 0, 2]
    GAS_in_Emul= c[:, :, 1, 2]
    GAS_in_Part = c[:, :, 2, 2]
    CHAR_in_Bub = c[:, :, 0, 3]#No transfer to this phase
    CHAR_in_Emul= c[:, :, 1, 3]#No transfer to this phase
    CHAR_in_Part = c[:, :, 2, 3]
    WAT_in_Bub = c[:, :, 0, 4]#No transfer to this phase
    WAT_in_Emul = c[:, :, 1, 4]#No transfer to this phase
    WAT_in_Part = c[:, :, 2, 4] #No transfer to this phase
    WATV_in_Bub = c[:, :, 0, 5]#No  transfer to this phase
    WATV_in_Emul = c[:, :, 1, 5]#No transfer to this phase
    WATV_in_Part = c[:, :, 2, 5]#No           

    RX_GAS_in_Bub = params["K4"] * TAR_in_Bub
    RX_CHAR_in_Bub = params["K5"] * TAR_in_Bub
    RX_GAS_in_Emul = params["K4"] * TAR_in_Emul
    RX_CHAR_in_Emul = params["K5"] * TAR_in_Emul

    MT_TAR_BE = params["Kbe"] * (TAR_in_Bub - TAR_in_Emul)
    MT_GAS_BE = params["Kbe"] * (GAS_in_Bub - GAS_in_Emul)
    MT_WATV_BE = params["Kbe"] * (WATV_in_Bub - WATV_in_Emul)

    # bubble balances
    s[:, :, 0, 1] =   - MT_TAR_BE - RX_GAS_in_Bub - RX_CHAR_in_Bub
    s[:, :, 0, 2] =   - MT_GAS_BE + RX_GAS_in_Bub
    s[:, :, 0, 3] =   - MT_GAS_BE + RX_CHAR_in_Bub
    s[:, :, 0, 5] =   - MT_WATV_BE

    # emulsion balances
    s[:, :, 1, 1] =    (MT_TAR_BE)*params["delta"]/params["ep_e"] - RX_GAS_in_Emul - RX_CHAR_in_Emul
    s[:, :, 1, 2] =   (MT_GAS_BE)*params["delta"]/params["ep_e"] + RX_GAS_in_Emul
    s[:, :, 1, 3] = RX_CHAR_in_Emul
    s[:, :, 1, 5] =   (MT_WATV_BE)*params["delta"]/params["ep_e"]
    
    return s

if __name__ == "__main__":
    main()