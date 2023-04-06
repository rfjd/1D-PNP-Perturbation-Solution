from math import *
from cmath import sqrt, tanh, exp
import matplotlib.pyplot as plt
import numpy as np
from pylatexenc.latex2text import LatexNodes2Text
from fractions import Fraction


# ************************************************************* #
# ************************ USER INPUTS ************************ #
# ************************************************************* #

# saving the input values for beta, gamma, Omega, and kl from the file Inputs.txt in a list
# dl_parameters: dimensionless parameters list [beta, gamma, Omega, kl]
# ind: index of the varying parameter (0: beta, 1: gamma, 2: Omega)
# pm: total number of values of the varying parameter (beta, gamma, or Omega)
def user_input():
    pm = 1
    ind = 0

    n_dlp = 4   # total number of dimensionless parameters (4: beta, gamma, Omega, kl)
    dl_parameters = [[0.0]]*n_dlp   # initial allocation of the dimensionless parameters list

    # reading the file into read_list
    file_id = open("Inputs.txt", "r")
    read_list = file_id.readlines()
    skip_ln = read_list.index('Parameters:\n')

    # storing the read_list information into the dl_parameters list
    # errors will be displayed in case of invalid inputs
    str_exist = ['param_beta', 'param_gamma', 'param_Omega', 'param_kl']
    exist_check = [False, False, False, False]
    num_var_param = 0
    for n in range(n_dlp):
        for elem in read_list[skip_ln:]:
            if str_exist[n] in elem:
                exist_check[n] = True
                if '(' in elem:
                    if n == 3:
                        print('INVALID INPUT FOR ' + str_exist[n] +
                              '. Single value is accepted! Follow the instructions in Inputs.txt.\nProgram Stopped.')
                        exit()
                    try:
                        ind_l = elem.index('(')
                        ind_h = elem.index(')')
                        if len(elem[ind_l+1:ind_h-1].split(',')) != 3:
                            print('INVALID INPUT FOR ' + str_exist[n] +
                                  '! Follow the instructions in Inputs.txt.\nProgram Stopped.')
                            exit()

                        syn = 'np.linspace' + elem[ind_l:]
                        dl_parameters[n] = eval(syn)
                        num_var_param += 1
                        ind = n
                        pm = len(dl_parameters[n])
                    except (TypeError, ValueError, SyntaxError):
                        print('INVALID INPUT FOR ' + str_exist[n] +
                              '! Follow the instructions in Inputs.txt.\nProgram Stopped.')
                        exit()
                elif '[' in elem:
                    if n == 3:
                        print('INVALID INPUT FOR ' + str_exist[n] +
                              '. Single value is accepted! Follow the instructions in Inputs.txt.\nProgram Stopped.')
                        exit()
                    try:
                        ind_l = elem.index('[')
                        syn = elem[ind_l:]
                        dl_parameters[n] = eval(syn)
                        num_var_param += 1
                        ind = n
                        pm = len(dl_parameters[n])
                    except (TypeError, ValueError, SyntaxError):
                        print('INVALID INPUT FOR ' + str_exist[n] +
                              '! Follow the instructions in Inputs.txt.\nProgram Stopped.')
                        exit()
                else:
                    if ',' in elem:
                        print('INVALID INPUT FOR ' + str_exist[n] +
                              '! Follow the instructions in Inputs.txt.\nProgram Stopped.')
                        exit()
                    try:
                        dl_parameters[n] = [eval(elem[elem.index('=')+1:])]
                    except (TypeError, ValueError, SyntaxError):
                        print('INVALID INPUT FOR ' + str_exist[n] +
                              '! Follow the instructions in Inputs.txt.\nProgram Stopped.')
                        exit()

    for i in range(n_dlp):
        if not exist_check[i]:
            print('INVALID INPUT! Parameter name(s) is wrong' +
                  '! Follow the instructions in Inputs.txt.\nProgram Stopped.')
            exit()

    if num_var_param > 1:
        print('INVALID INPUT! There are more than one varying parameter' +
              '! Follow the instructions in Inputs.txt.\nProgram Stopped.')
        exit()

    # updating the pm value
    for i in range(len(dl_parameters)):
        if i != ind:
            dl_parameters[i] *= pm

    err_text = r'INVALID INPUT! You need $\mid\beta\mid\le 1$ (beta),'\
               r'$\gamma\:\in\:[-1/5 -1/3 -1/2 0 1/2 1/3 1/5]$ (gamma),' \
               r'$\Omega>0$ (Omega), $\Phi_0>0$ (Phi0), $\kappa\ell>0$ (L).\\Program Stopped.'
    bool_beta = 1 >= max(dl_parameters[0][:]) and -1 <= min(dl_parameters[0][:])
    bool_gamma = True
    for i in range(pm):
        if dl_parameters[1][i] not in [-1/5, -1/3, -1/2, 0, 1/2, 1/3, 1/5]:
            bool_gamma = False
    bool_omega = 0 < min(dl_parameters[2][:])
    bool_kl = 0 < min(dl_parameters[3][:])

    if bool_beta is False or bool_gamma is False or bool_omega is False or bool_kl is False:
        print(LatexNodes2Text().latex_to_text(err_text))
        exit()

    file_id.close()
    return [dl_parameters, [ind, pm]]


# ************************************************************** #
# **************** DISCRETIZATION OF THE DOMAIN **************** #
# ************************************************************** #

# x: vector of the grid points (face-centered) from -kl to kl
# h: vector of the grid spacings (i.e., h[i] = x[i+1] - x[i])
# im: total number of grids
# param_kl: electrode spacing
# uniform/is_uniform: boolian, default value = False
# h_min/h_minimum: minimum grid spacing, default value = 0.01
# r/ref: powerlaw growth rate of the grid spacing, default value = 1.01
# discretization parameters
is_uniform = False
h_minimum = 0.01
ref = 1.01


# stretched grid constructor
def stretched_grid(param_kl, h_min, r):
    n = int(floor(log(1-(1-r)*(param_kl/h_min))/log(r))+1)
    h = np.array([0]*2*n, dtype=float)
    h[0] = h_min
    for i in range(1, n):
        h[i] = r*h[i-1]
    if np.sum(h[0:n]) > param_kl:
        h[n-1] = param_kl - np.sum(h[0:n-1])

    h[n:2*n] = np.flip(h[0:n])
    im = 2*n+1
    x = np.array([0.0]*im, dtype=float)
    x[0] = -param_kl
    for i in range(1, 2*n+1):
        x[i] = x[i-1] + h[i-1]
    return [x, h, im]


# grid and spacing generator
def domain(param_kl, uniform, h_min, r):
    if uniform is True:
        # unform grid spacing of size h_min
        deb_im = 2**(-floor(log(h_min, 2)))
        im = int(floor(2*param_kl*deb_im)+1)
        h = [2*param_kl/(im-1)]*(im-1)
        x = np.array([-param_kl+i*h[0] for i in range(im)], dtype=float)
    else:
        [x, h, im] = stretched_grid(param_kl, h_min, r)

    return [x, h, im]


# ************************************************************** #
# ************* NUMERICAL SOLUTION TO THE AREF EQN ************* #
# ************************** (eqn 49) ************************** #
# ************************************************************** #


# semi-analytical second-order approximation to AREF
# eqn numbers refer to the corresponding equations in ref. 1.
def semi_analytical_aref(param_beta, param_gamma, param_Omega, param_kl, x, h, im):
    # rhs of the AREF eqn
    rhs = np.array([0] * im, dtype=complex)
    sc_checker = 4*(param_beta*param_Omega)**2
    if param_gamma == 0 and sc_checker == 1:  # Special case (Appendix A)
        sgn = np.sign(param_beta)  # S (eqn A4b)
        lambda12 = sqrt((1+2j*param_Omega)/2)  # \lambda (eqn A4a)
        cap_gamma = 2*sgn*param_kl*(4*(lambda12**4-lambda12**2+1)*exp(-2*lambda12*param_kl) +
                                    (2*lambda12**4-2*lambda12**2+1)*(1+exp(-4*lambda12*param_kl)) +
                                    (4*lambda12**2-3)*(1-exp(-4*lambda12*param_kl))/(2*lambda12*param_kl))
        for i in range(im):
            # A\sinh(\lambda\tilde{x})
            a_sinh = lambda12/cap_gamma*(((4*lambda12**2-1)*sgn+1j)*(1+exp(-2*lambda12*param_kl)) -
                                         (sgn-1j)*lambda12*param_kl*(1-exp(-2*lambda12*param_kl)))*(
                                          exp(lambda12*(x[i]-param_kl))-exp(-lambda12*(x[i]+param_kl)))
            # B\sinh(\lambda\tilde{x})
            b_sinh = lambda12/cap_gamma*(((4*lambda12**2-1)*1j+sgn)*(1+exp(-2*lambda12*param_kl)) +
                                         (sgn-1j)*lambda12*param_kl*(1-exp(-2*lambda12*param_kl)))*(
                                          exp(lambda12*(x[i]-param_kl))-exp(-lambda12*(x[i]+param_kl)))
            # (A+B)\cosh(\lambda\tilde{x})
            apb_cosh = 4*lambda12**3*(sgn+1j)/cap_gamma*(1+exp(-2*lambda12*param_kl))*(
                       exp(lambda12*(x[i]-param_kl))+exp(-lambda12*(x[i]+param_kl)))
            # (A+B)\sinh(\lambda\tilde{x})
            apb_sinh = 4*lambda12**3*(sgn+1j)/cap_gamma*(1+exp(-2*lambda12*param_kl))*(
                       exp(lambda12*(x[i]-param_kl))-exp(-lambda12*(x[i]+param_kl)))
            # (A-iSB)\cosh(\lambda\tilde{x})
            amisb_cosh = lambda12/cap_gamma*((2*sgn*(4*lambda12**2-1)+1j*(1-sgn**2))*(1+exp(-2*lambda12*param_kl)) +
                                             (1j*(1-sgn**2)-2*sgn)*lambda12*param_kl*(1-exp(-2*lambda12*param_kl)))*(
                                              exp(lambda12*(x[i]-param_kl))+exp(-lambda12*(x[i]+param_kl)))
            # C
            c = -2*sgn/cap_gamma*(2*lambda12**4-2*lambda12**2+1)*(1+exp(-2*lambda12*param_kl))**2

            # \hat{n}_+^{(1)}: eqn A1
            n1_hat = a_sinh-1j*sgn/(4*lambda12)*x[i]*apb_cosh
            # \hat{n}_-^{(1)}: eqn A2
            n2_hat = 1j*sgn*b_sinh-1/(4*lambda12)*x[i]*apb_cosh
            # \hat{E}^{(1)}: -derivative of eqn A3 (the electric potential \hat{\phi}^{(1)})
            e_hat = -(c-amisb_cosh/(2*lambda12)-(1j*sgn-1)/(8*lambda12**3)*apb_cosh +
                      (1j*sgn-1)/(8*lambda12**2)*x[i]*apb_sinh)
            # complex conjugates: \bar{X}=\mathrm{conj}{\hat{X}}
            n1_bar = np.conj(n1_hat)
            n2_bar = np.conj(n2_hat)
            e_bar = np.conj(e_hat)
            # rhs of the AREF eqn: eqn 50
            rhs[i] = 1/8*((n1_hat+n2_hat)*e_bar+(n1_bar+n2_bar)*e_hat)
    else:
        cap_delta = 1-4*param_beta*param_Omega*(1j*param_gamma+param_beta*param_Omega)  # \Delta (eqn 32b)
        lambda1 = sqrt((1+2j*param_Omega+sqrt(cap_delta))/2)  # \lambda_+ (eqn 33)
        lambda2 = sqrt((1+2j*param_Omega-sqrt(cap_delta))/2)  # \lambda_- (eqn 33)
        s = 2j*param_beta*param_Omega+sqrt(cap_delta)  # s (eqn 32a)
        # Gamma: eqn 37
        cap_gamma = s**2-2*param_gamma*s+1-1/(2*param_kl)*(
                (param_gamma+1)*(s-1)**2*(lambda2*param_kl-tanh(lambda2*param_kl))/(lambda2**3) -
                (param_gamma-1)*(s+1)**2*(lambda1*param_kl-tanh(lambda1*param_kl))/(lambda1**3))
        for i in range(im):
            c1 = (exp(lambda1*(x[i]-param_kl))-exp(-lambda1*(x[i]+param_kl)))/(1+exp(-2*lambda1*param_kl))
            c2 = (exp(lambda2*(x[i]-param_kl))-exp(-lambda2*(x[i]+param_kl)))/(1+exp(-2*lambda2*param_kl))
            # \hat{n}_+^{(1)}: eqn 30
            n1_hat = 1/cap_gamma*((s-param_gamma)*(s-1)/(lambda2*param_kl)*c2
                                  + (1-param_gamma)*(s+1)/(lambda1*param_kl)*c1)
            # \hat{n}_-^{(1)}: eqn 31
            n2_hat = 1/cap_gamma*((1+param_gamma)*(s-1)/(lambda2*param_kl)*c2
                                  - (s-param_gamma)*(s+1)/(lambda1*param_kl)*c1)

            c1p = (exp(lambda1*(x[i]-param_kl))+exp(-lambda1*(x[i]+param_kl)))/(1+exp(-2*lambda1*param_kl))
            c2p = (exp(lambda2*(x[i]-param_kl))+exp(-lambda2*(x[i]+param_kl)))/(1+exp(-2*lambda2*param_kl))
            # \hat{E}^{(1)}: -derivative of eqn 34 (the electric potential \hat{\phi}^{(1)})
            e_hat = 1/param_kl+1/(2*param_kl*cap_gamma)*(
                    (1+param_gamma)*(s-1)**2/(lambda2**2)*(c2p-tanh(lambda2*param_kl)/(lambda2*param_kl)) +
                    (1-param_gamma)*(s+1)**2/(lambda1**2)*(c1p-tanh(lambda1*param_kl)/(lambda1*param_kl)))
            # complex conjugates: \bar{X}=\mathrm{conj}{\hat{X}}
            n1_bar = np.conj(n1_hat)
            n2_bar = np.conj(n2_hat)
            e_bar = np.conj(e_hat)

            # rhs of the AREF eqn: eqn 50
            rhs[i] = 1/8*(((1+param_gamma)**2*n1_hat + (1-param_gamma)**2*n2_hat)*e_bar + (
                    (1+param_gamma)**2*n1_bar + (1-param_gamma)**2*n2_bar)*e_hat)
    # constructing the linear system of algebraic equations (cf. Appendix B of ref. 1, eqn B4)
    # a: coefficient matrix
    # b: right hand side column vector
    a = np.zeros((im, im), dtype=float)
    b = np.transpose(rhs)

    for i in range(1, im-1):
        a[i, i-1] = 2/(h[i-1])/(h[i] + h[i-1])             # eqn B3a
        a[i, i] = -(1+2*(1/h[i]+1/h[i-1])/(h[i]+h[i-1]))   # eqn B3b
        a[i, i+1] = 2/(h[i])/(h[i]+h[i-1])                 # eqn B3c

    # computing AREF
    aref_a = np.array([0] * im, dtype=complex)
    aref_a[1:im-1] = np.linalg.solve(a[1:im-1, 1:im-1], b[1:im-1])

    return np.real(aref_a)


# *************************************************************** #
# *********************** WRITE INTO FILE *********************** #
# *************************************************************** #
param_string_latex = [r'$\beta$', r'$\gamma$', r'$\Omega$', r'$\kappa\ell$', r'$\Phi_0$']


# converting numbers into strings
def convert_to_string(val):
    if int(val) == val:
        string = str(int(val))
    else:
        order = log(abs(val), 10)
        if order > 0:
            string = str(round(val, 4))
        else:
            string = str(round(val, int(abs(floor(order))) + 3))

    return string


# writing the results (x, AREF) into a 2-column .txt file
def write_to_file(aref_a, x, dl_params, param_string):
    data = np.column_stack((x, aref_a))
    filename = 'results_beta=' + convert_to_string(dl_params[0]) + '_gamma=' + convert_to_string(dl_params[1]) \
               + '_Omega=' + convert_to_string(dl_params[2]) + '_kl=' + convert_to_string(dl_params[3]) + '.txt'
    file_id = open(filename, "w", encoding="utf-8")
    message = r'AREF_CALC results: $\kappa x$ (1st column) vs $\langle\tilde{E}\rangle/\Phi_0^2$ (2nd column)'
    file_id.write(LatexNodes2Text().latex_to_text(message))
    file_id.write('\n' + '='*50 + '\n')
    for i in range(len(dl_params)):
        param_message = param_string[i] + r' $=$ ' + convert_to_string(dl_params[i])
        file_id.write(LatexNodes2Text().latex_to_text(param_message))
        file_id.write('\n')

    file_id.write('='*50 + '\n')
    np.savetxt(file_id, data, fmt=['%+10.16f', '%10.16f'])
    file_id.close()

    return 0


# ************************************************************** #
# ************************** PLOTTING ************************** #
# ************************************************************** #

# plot options
app_params = {'figure.dpi': 100,
              'font.size': 12,
              'mathtext.fontset': 'stix',
              'font.family': 'STIXGeneral',
              'lines.linewidth': 2,
              'xtick.direction': 'in',
              'ytick.direction': 'in',
              'xtick.top': True,
              'ytick.right': True,
              'figure.figsize': [7, 5]}
plt.rcParams.update(app_params)
plt.subplots_adjust(left=0.15, bottom=0.1, right=0.8, top=0.9)
l_d_critical = 0.32


# constructing the plot's legend
def legend(ind, pm, dl_parameters):
    dl = dl_parameters[ind]
    legend_label = [' ']*pm
    for i in range(pm):
        legend_label[i] = convert_to_string(dl[i])

    return legend_label


# plot appearance
def plot_app(param_kl, param_string, ind, y_max, pm):
    if y_max == 0.0:
        scale = 10**(-16)
    else:
        scale = 10**(round(log(y_max, 10)))

    plt.xlabel(r'$\kappa x$')
    power = int(log(1/scale, 10))
    if power == 1:
        plt.ylabel(r'$\langle\tilde{E}\rangle/\Phi_0^2 \times 10$')
    elif power == 0:
        plt.ylabel(r'$\langle\tilde{E}\rangle/\Phi_0^2$')
    else:
        plt.ylabel(r'$\langle\tilde{E}\rangle/\Phi_0^2 \times 10^{' + str(power) + '}$')

    plt.xlim(-param_kl, param_kl)

    rat = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y_lim = rat[rat.index(int(floor(y_max/scale)))]+1
    plt.ylim(-y_lim*scale, y_lim*scale)
    yticks = [-y_lim*scale, -y_lim*scale/2, 0, y_lim*scale/2, y_lim*scale]
    yticklabels = [' ']*5
    for i in range(5):
        yticklabels[i] = '$' + str(round(yticks[i]/scale, 1)) + '$'

    plt.yticks((-y_lim*scale, -y_lim*scale/2, 0, y_lim*scale/2, y_lim*scale),
               (yticklabels[0], yticklabels[1], yticklabels[2], yticklabels[3], yticklabels[4]))
    if pm > 1:
        plt.legend(title=param_string[ind], fancybox=False, edgecolor='w', bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.show()


# calculating the AREF length scale (eqn 53)
def l_aref(kl_vec, omega_vec, pm):
    l_d_vec = [0.0]*pm
    for i in range(pm):
        l_d_vec[i] = (1/(omega_vec[i]*kl_vec[i]**2))**(1/2)

    return np.min(l_d_vec)


# finding the peak AREF magnitude to be used for the ylim
def y_max_finder(x, param_kl, aref_a, l_d, l_d_c):
    if l_d > l_d_c:
        y_max = np.max(abs(aref_a))
    else:
        ind_micronscale = np.min(np.nonzero(x + param_kl > 2))
        ind_midplane = int((len(x)-1)/2)
        y_max_p = np.max(aref_a[ind_micronscale:ind_midplane])
        y_max_m = np.min(aref_a[ind_micronscale:ind_midplane])
        if y_max_p > abs(y_max_m):
            y_max = y_max_p
        else:
            y_max = -y_max_m

    return y_max
