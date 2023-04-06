"""
********************************************************************************* +
*********************************** AREF_CALC *********************************** +
************************** S.M.H. (Aref) Hashemi Amrei ************************** +
********************************************************************************* +
This program evaluates AREF via a second-order perturbation solution to the PNP eqns.
AREF: asymmetric rectified electric field

References to cite:
    1) A perturbation solution to the full Poisson–Nernst–Planck equations yields an asymmetric rectified electric field
    S. M. H. Hashemi Amrei, Gregory H. Miller, Kyle J. M. Bishop, & William D. Ristenpart
    Soft Matter 16, 7052-7062 (2020)

    2) Oscillating Electric Fields in Liquids Create a Long-Range Steady Field
    S. M. H. Hashemi Amrei, Scott C. Bukosky, Sean P. Rader, William D. Ristenpart, & Gregory H. Miller
    Physical Review Letters 121, 185504 (2018)

The eqn numbers throughout this code refer to the corresponding equations in ref. 1.
___________________________________________________________________________________
Correspondence should be addressed to:
    i) S.M.H. (Aref) Hashemi Amrei (aref@cims.nyu.edu)
    ii) Gregory H. Miller (grgmiller@ucdavis.edu)
    iii) William D. Ristenpart (wdristenpart@ucdavis.edu)
"""

from AREF_SOURCE import *

print('\n============================== AREF_CALC by S.M.H. Hashemi Amrei (Aref) ==============================\n' +
      'This program evaluates a semi-analytical AREF (asymmetric rectified electric field).\n\n' +
      'References to cite:\n' +
      '1) A perturbation solution to the full Poisson–Nernst–Planck equations yields an asymmetric rectified' +
      ' electric field\nS. M. H. Hashemi Amrei, Gregory H. Miller, Kyle J. M. Bishop, & William D. Ristenpart\n' +
      'Soft Matter, DOI: 10.1039/D0SM00417K (2020)\n' +
      '2) Oscillating Electric Fields in Liquids Create a Long-Range Steady Field\n' +
      'S. M. H. Hashemi Amrei, Scott C. Bukosky, Sean P. Rader, William D. Ristenpart, & Gregory H. Miller\n' +
      'Physical Review Letters 121, 185504 (2018)\n' +
      '______________________________________________________________________\n' +
      'Correspondence should be addressed to:\n' +
      'i) S.M.H. (Aref) Hashemi Amrei (aref@cims.nyu.edu)\n' +
      'ii) Gregory H. Miller (grgmiller@ucdavis.edu)\n' +
      'iii) William D. Ristenpart (wdristenpart@ucdavis.edu)\n' +
      '======================================================================================================\n')

print('Program started... ')

# saving the input values for beta, gamma, Omega, and kl from the file Inputs.txt in a list
[dl_parameters, [ind, pm]] = user_input()
# grid and spacing generator
[x, h, im] = domain(dl_parameters[3][0], is_uniform, h_minimum, ref)
# preallocation of the legend list (plotting)
legendLabel = legend(ind, pm, dl_parameters)
# preallocation of the AREF peak magnitude vector (plotting)
y_max_vec = np.zeros(pm, dtype=float)
# calculating the AREF length scale (plotting)
len_d = l_aref(dl_parameters[3], dl_parameters[2], pm)
# evalauting the AREF for different set of dimensionless parameters (a total of pm sets) and plotting the results
for i in range(pm):
    # extracting the dimensionless parameters from the list
    [beta, gamma, Omega, kl] = [dl_parameters[0][i], dl_parameters[1][i], dl_parameters[2][i], dl_parameters[3][i]]
    # semi-analytical second-order approximation to AREF
    AREF_A = semi_analytical_aref(beta, gamma, Omega, kl, x, h, im)
    # finding the peak AREF magnitude to be used for the ylim (plotting)
    y_max_vec[i] = y_max_finder(x, kl, AREF_A, len_d, l_d_critical)
    # plotting the AREF distribution
    plt.plot(x, AREF_A, label=legendLabel[i])
    # writing the results (x, AREF) into a 2-column .txt file
    write_to_file(AREF_A, x, [beta, gamma, Omega, kl], param_string_latex)

# plot appearance (plotting)
print('Program finished.')
plot_app(dl_parameters[3][0], param_string_latex, ind, np.max(y_max_vec), pm)


