# Author:  Ye Li <pkupkupku@pku.edu.cn>
# Updated: 2025-4-10

# Main reference: [Y. Mei et al, J. Phys. Chem. A 123, 666, 2018.] 

from functools import reduce
import pyscf
import numpy as np
from pyscf import gto, scf, dft
from pyscf.lib import logger


def orb_correspondence(mf,key='beta'):
    """
    For each beta spatial orbital, find the most similar alpha spatial orbital.

    Parameters:
    ----------
    mf : pyscf.dft.UKS
        The UKS object containing the molecular orbital coefficients.
        The molecular orbital coefficients are stored in the `mo_coeff` attribute.
        The overlap matrix is obtained using the `get_ovlp()` method.

    Returns:
    -------
    orb_dict : dict
        A dictionary mapping the index of each beta(alpha) orbital 
        to the index of the most similar alpha(beta) orbital.
    similarity : dict
        A dictionary mapping the index of each beta(alpha) orbital 
        to the similarity value with the corresponding alpha(beta) orbital.
    """
    mo_coeff    =   mf.mo_coeff
    ovlp        =   mf.get_ovlp()
    if key == 'beta':
        S_ab        =   reduce(np.dot, (mo_coeff[0].conj().T, ovlp, mo_coeff[1]))
    elif key == 'alpha':
        S_ab        =   reduce(np.dot, (mo_coeff[1].conj().T, ovlp, mo_coeff[0]))
    else:
        raise ValueError('key must be "alpha" or "beta".')
    orb_dict    =   {}
    similarity  =   {}
    
    for i in range(S_ab.shape[1]):
        # Find the index of the maximum absolute value in the i-th column
        orb_dict[i] = np.argmax(abs(S_ab[:, i]))
        # Store the similarity value
        similarity[i] = abs(S_ab[orb_dict[i], i])
    return orb_dict, similarity


def calculate_spin(mf, mo_occ=None, check=False, target_spin=None, verbose=None):
    """
    Check if the spin of the system is equal to the target spin.

    Parameters:
    ----------
    mf : pyscf.dft.UKS
        The UKS object containing the molecular orbital coefficients.
    mo_occ : list of numpy.ndarray, optional
        The occupation numbers of the molecular orbitals.
        If None, it is obtained from `mf.mo_occ`.
    target_spin : float (of a half_int), optional
        The target spin value to check against. 
        If None, it is calculated based on the number of electrons.
    verbose : int, optional
        The verbosity level for logging. 
        If None, it uses the verbosity level of `mf`.

    Returns:
    -------
    ss : float
        The spin square S(S+1) of the system.
    s : float
        The spin multiplicity 2S+1 of the system.
    """
    mo_coeff    =   mf.mo_coeff
    if mo_occ is None:
        mo_occ      =   mf.mo_occ

    # Calculate the current spin square and spin multiplicity
    ss, s       =   mf.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                                    mo_coeff[1][:,mo_occ[1]>0]), mf.get_ovlp())
    
    if not check:
        return ss, s
    else:
        # Calculate the target spin and spin square
        if target_spin is None:
            n_a         =   np.sum(mo_occ[0])
            n_b         =   np.sum(mo_occ[1])
            target_spin =   0.5 * abs(n_a - n_b)
        target_ss   =   target_spin * (target_spin + 1)
        target_s    =   target_spin * 2 + 1
        # Output the check results
        if verbose is None:
            verbose = mf.verbose
        log   =   logger.new_logger(mf, verbose)
        if (abs(target_s - s) > 0.05*max(target_s, 1) 
            or abs(target_ss - ss) > 0.05*max(target_ss, 1)):
            log.warn('''Strong spin contamination detecte!
                    The QE-DFT result might be unreliable!''')
            log.warn('Spin square <S^2> mismatch: %s != %s (expected)', 
                    ss, target_spin*(target_spin+1))
            log.warn('Spin multiplicity 2S+1 mismatch: %s != %s (expected)', 
                    s, target_spin*2+1)
        elif log.verbose >= logger.INFO:
            log.info('Target <S^2>: %s, and 2S+1: %s', target_ss, target_s)
            log.info('Current <S^2>: %s, and 2S+1: %s', ss, s)
        return ss, s
    

class QEDFT_Doublet_Base():
    """
    Methods for doublet (N-1) systems.
    """
    def __init__(self, mf):
        self.mf = mf
        pass

    def triplet_states(self, mf=None, nstates=3):
        """
        Placeholder for triplet states calculation.
        """
        if mf is None:
            mf = self.mf

        n_alpha, _ = mf.nelec
        nstates = min(nstates, len(mf.mo_occ[0])-n_alpha)

        energies = []
        orbitals = []
        spins = []
        for i in range(nstates):
            energies.append(mf.mo_energy[0][n_alpha + i])
            orbitals.append(n_alpha + i) 
            mo_occ_T = 1 * mf.mo_occ
            mo_occ_T[0][n_alpha + i] = 1 # Add one alpha electron
            ss, s = calculate_spin(mf, mo_occ=mo_occ_T)
            spins.append((ss,s))
        return energies, orbitals, spins


    def singlet_states(self, mf=None, purification='h', nstates=3):
        """
        Placeholder for singlet states calculation.
        """
        if mf is None:
            mf = self.mf

        n_alpha, n_beta = mf.nelec
        nstates       = min(nstates, len(mf.mo_occ[1])-n_beta)
        nstates_trial = min(max(nstates+3, nstates*2), len(mf.mo_occ[1])-n_beta)
        purification = purification.lower()[0]
        if purification not in ['h', 'y']:
            raise ValueError('Purification type not supported.')
        
        orb_dict, similarity = orb_correspondence(mf)

        energies = []
        orbitals = []
        spins    = []
        sims     = []
        
        for i in range(nstates_trial):
            # Calculate the energy and spin of the symmetry-broken state
            e_BS  = mf.mo_energy[1][n_beta + i] 
            mo_occ_BS = 1 * mf.mo_occ
            mo_occ_BS[1][n_beta + i] = 1 # Add one beta electron
            ss_BS, s_BS = calculate_spin(mf, mo_occ=mo_occ_BS)

            # Find the corresponding triplet state,
            orb_T = orb_dict[n_beta + i]
            similarity_T = similarity[n_beta + i]
            if orb_T <= n_alpha - 1:
                # If the most similar alpha orbital is already occupied, 
                # then there is NO corresponding triplet state,
                # and we assume the broken-symmetry state is 100% singlet.
                e_S = e_BS
                ss_S = ss_BS
            else:
                e_T = mf.mo_energy[0][orb_T]
                mo_occ_T = 1 * mf.mo_occ
                mo_occ_T[0][orb_T] = 1
                ss_T, _ = calculate_spin(mf, mo_occ=mo_occ_T)
                if purification == 'h': 
                    # Half-Half spin purification.
                    # We assume the broken-symmetry state is 50%-50% singlet-triplet.
                    ss_S = 2*ss_BS - ss_T
                    e_S  = 2*e_BS - e_T
                elif purification == 'y':
                    # Yamaguchi spin purification.
                    # We force <S^2>_S = <S^2>_T - 2.
                    ss_S = ss_T - 2
                    c = (ss_T - ss_BS) / 2
                    e_S = (1/c) * e_BS + (1-1/c) * e_T
                else:
                    raise ValueError('Purification type not supported.')
            s_S = 2 * np.sqrt(ss_S + 0.25)
            energies.append(e_S)
            orbitals.append(n_beta + i)
            spins.append((ss_S, s_S))
            sims.append((orb_T, similarity_T))

        # Reorder the energies, orbitals, spins and similarities
        sorted_lists = sorted(zip(energies, orbitals, spins, sims), key=lambda x: x[0])
        energies, orbitals, spins, sims = zip(*sorted_lists)
        # Keep only the first nstates elements
        energies = list(energies)[:nstates]
        orbitals = list(orbitals)[:nstates]
        spins    = list(spins)[:nstates]
        sims     = list(sims)[:nstates]
        return energies, orbitals, spins, sims
    
    def kernel(self, mf=None, purification='h', nstates=3):

        if mf is None:
            mf = self.mf
        # Check the spin of the (N-1) system
        _, _ = calculate_spin(mf, check=True)

        # Calculate all the singlet and triplet states of (N) system
        energies_S, orbitals_S, spins_S, sims_S = self.singlet_states(mf, purification=purification, nstates=nstates)
        energies_T, orbitals_T, spins_T = self.triplet_states(mf, nstates=nstates)

        # Find the ground state of the (N) system
        self.gs_energy   =   energies_S[0]
        self.gs_orbital  =   orbitals_S[0] + 1
        self.gs_spin     =   'Singlet'
        self.gs_ss       =   spins_S[0][0]
        self.gs_s        =   spins_S[0][1]
        if energies_T[0] < self.gs_energy:
            self.gs_energy   =   energies_T[0]
            self.gs_orbital  =   orbitals_T[0] + 1
            self.gs_spin     =   'Triplet'
            self.gs_ss       =   spins_T[0][0]
            self.gs_s        =   spins_T[0][1]

        # Restore all excited states
        if self.gs_spin == 'Singlet':
            self.singlet_energy  = [(x-self.gs_energy)*27.2114 for x in energies_S[1:]]
            self.singlet_orbital = [x + 1 for x in orbitals_S[1:]]
            self.singlet_spin    = spins_S[1:]
            self.singlet_sim     = sims_S[1:]
            self.triplet_energy  = [(x-self.gs_energy)*27.2114 for x in energies_T]
            self.triplet_orbital = [x + 1 for x in orbitals_T]
            self.triplet_spin    = spins_T
        else:
            self.singlet_energy  = [(x-self.gs_energy)*27.2114 for x in energies_S]
            self.singlet_orbital = [x + 1 for x in orbitals_S]
            self.singlet_spin    = spins_S
            self.singlet_sim     = sims_S
            self.triplet_energy  = [(x-self.gs_energy)*27.2114 for x in energies_T[1:]]
            self.triplet_orbital = [x + 1 for x in orbitals_T[1:]]
            self.triplet_spin    = spins_T[1:]
        
        # update the ground state energy
        self.gs_energy = mf.e_tot + self.gs_energy
        
        # Print the excitation energies
        if mf.verbose >= logger.QUIET:
            log = logger.new_logger(mf, mf.verbose)
            log.log('###############################################################')
            log.log('##########         QEDFT Excited Energies          ############')
            log.log('###############################################################')
            log.log('Singlet excited energies (in eV):')
            log.log(", ".join([str(x) for x in self.singlet_energy]))
            log.log('Triplet excited energies (in eV):')
            log.log(", ".join([str(x) for x in self.triplet_energy]))
            log.log('')

        return self.singlet_energy, self.triplet_energy
    
    def analyze(self, verbose=None):
        """
        Placeholder for the analyze function.
        """
        # Print all information of excited states
        if verbose is None:
            verbose = self.mf.verbose
        log = logger.new_logger(self.mf, verbose)
        if log.verbose >= logger.NOTE:
            log.note('###############################################################')
            log.note('##############         QEDFT Analysis          ################')
            log.note('###############################################################')
            log.note('')
            log.note('The (N) system has a %s ground state', self.gs_spin)
            log.note('The ground state has spin <S^2>=%.6f, and 2S+1=%.6f', self.gs_ss, self.gs_s)
            log.note('')

        if log.verbose >= logger.NOTE:
            log.note('Singlet excited states:')
            for i in range(len(self.singlet_energy)):
                log.note('State %d: E = %.6f eV, <S^2> = %.6f, 2S+1 = %.6f',
                        i+1, self.singlet_energy[i],
                        self.singlet_spin[i][0], self.singlet_spin[i][1]) 
                if self.gs_spin == 'Singlet':
                    log.note('      %db --> %db', self.gs_orbital, self.singlet_orbital[i])
                else:
                    log.note('      %da --> %db', self.gs_orbital, self.singlet_orbital[i])
                log.note('      Orbital similarity: (%da, %.6f)', 
                        self.singlet_sim[i][0], self.singlet_sim[i][1])
            log.note('')    
            log.note('Triplet excited states:')
            for i in range(len(self.triplet_energy)):
                log.note('State %d: E = %.6f eV, <S^2> = %.6f, 2S+1 = %.6f',
                        i+1, self.triplet_energy[i],
                        self.triplet_spin[i][0], self.triplet_spin[i][1]) 
                if self.gs_spin == 'Singlet':
                    log.note('      %db --> %da', self.gs_orbital, self.triplet_orbital[i])
                else:
                    log.note('      %da --> %da', self.gs_orbital, self.triplet_orbital[i])
            log.note('')
        return None
    
    def vertical_ST_gap(self):
        """
        Calculate the singlet-triplet gap.

        Returns:
        -------
        gap_type : str
            The type of the singlet-triplet gap.
        E_ST : float
            The singlet-triplet gap in eV.
            If the ground state is singlet, it is S1-T1 gap;
            If the ground state is triplet, it is S1-T0 gap.
        """

        if self.gs_spin == 'Singlet':
            # If the ground state is singlet, we calculate the S1-T1 gap:
            return 'S1-T1', self.singlet_energy[0] - self.triplet_energy[0]
        else:
            # If the ground state is triplet, we calculate the S1-T0 gap:
            return 'S1-T0', self.singlet_energy[0]


class QEDFT_Doublet_Base_Nplus1():
    """
    Methods for doublet (N+1) systems.
    """
    def __init__(self, mf):
        self.mf = mf
        pass

    def triplet_states(self, mf=None, nstates=3):
        """
        Placeholder for triplet states calculation.
        """
        if mf is None:
            mf = self.mf

        _, n_beta = mf.nelec
        nstates = min(nstates, n_beta)

        energies = []
        orbitals = []
        spins = []
        for i in range(nstates):
            energies.append(mf.mo_energy[1][n_beta - i - 1])
            orbitals.append(n_beta - i - 1) 
            mo_occ_T = 1 * mf.mo_occ
            mo_occ_T[1][n_beta - i - 1] = 0 # Remove a beta electron
            ss, s = calculate_spin(mf, mo_occ=mo_occ_T)
            spins.append((ss,s))
        return energies, orbitals, spins
    

    def singlet_states(self, mf=None, purification='h', nstates=3):
        """
        Placeholder for singlet states calculation.
        """
        if mf is None:
            mf = self.mf

        n_alpha, n_beta = mf.nelec
        nstates       = min(nstates, n_beta)
        nstates_trial = min(max(nstates+3, nstates*2), n_beta)
        purification = purification.lower()[0]
        if purification not in ['h', 'y']:
            raise ValueError('Purification type not supported.')
        
        orb_dict, similarity = orb_correspondence(mf,key='alpha') 
        # Now the key are indexes of alpha orbitals

        energies = []
        orbitals = []
        spins    = []
        sims     = []
        
        for i in range(nstates_trial):
            # Calculate the energy and spin of the symmetry-broken state
            e_BS  = mf.mo_energy[0][n_alpha - i - 1] # occ alpha orbital energy
            mo_occ_BS = 1 * mf.mo_occ
            mo_occ_BS[0][n_alpha - i - 1] = 0 # Remove an alpha electron
            ss_BS, s_BS = calculate_spin(mf, mo_occ=mo_occ_BS)

            # Find the corresponding triplet state,
            orb_T = orb_dict[n_alpha - i - 1]
            similarity_T = similarity[n_alpha - i - 1]
            if orb_T >= n_beta:
                # If the most similar beta orbital is already virtual, 
                # then there is NO corresponding triplet state,
                # and we assume the broken-symmetry state is 100% singlet.
                e_S = e_BS
                ss_S = ss_BS
            else:
                e_T = mf.mo_energy[1][orb_T]
                mo_occ_T = 1 * mf.mo_occ
                mo_occ_T[1][orb_T] = 0 # Remove the beta electron
                ss_T, _ = calculate_spin(mf, mo_occ=mo_occ_T)
                if purification == 'h': 
                    # Half-Half spin purification.
                    # We assume the broken-symmetry state is 50%-50% singlet-triplet.
                    ss_S = 2*ss_BS - ss_T
                    e_S  = 2*e_BS - e_T
                elif purification == 'y':
                    # Yamaguchi spin purification.
                    # We force <S^2>_S = <S^2>_T - 2.
                    ss_S = ss_T - 2
                    c = (ss_T - ss_BS) / 2
                    e_S = (1/c) * e_BS + (1-1/c) * e_T
                else:
                    raise ValueError('Purification type not supported.')
            s_S = 2 * np.sqrt(ss_S + 0.25)
            energies.append(e_S)
            orbitals.append(n_alpha - i - 1)
            spins.append((ss_S, s_S))
            sims.append((orb_T, similarity_T))

        # Reorder the energies, orbitals, spins and similarities
        # Note that now, higher orbitals have lower excitation energy
        sorted_lists = sorted(zip(energies, orbitals, spins, sims), 
                              key=lambda x: x[0], reverse=True)
        energies, orbitals, spins, sims = zip(*sorted_lists)
        # Keep only the first nstates elements
        energies = list(energies)[:nstates]
        orbitals = list(orbitals)[:nstates]
        spins    = list(spins)[:nstates]
        sims     = list(sims)[:nstates]
        return energies, orbitals, spins, sims
    
    def kernel(self, mf=None, purification='h', nstates=3):

        if mf is None:
            mf = self.mf
        # Check the spin of the (N+1) system
        _, _ = calculate_spin(mf, check=True)

        # Calculate all the singlet and triplet states of (N) system
        energies_S, orbitals_S, spins_S, sims_S = self.singlet_states(mf, purification=purification, nstates=nstates)
        energies_T, orbitals_T, spins_T = self.triplet_states(mf, nstates=nstates)

        # Find the ground state of the (N) system
        self.gs_energy   =   energies_S[0]
        self.gs_orbital  =   orbitals_S[0] + 1
        self.gs_spin     =   'Singlet'
        self.gs_ss       =   spins_S[0][0]
        self.gs_s        =   spins_S[0][1]
        if energies_T[0] > self.gs_energy: # higher orbital energy means lower excitation energy
            self.gs_energy   =   energies_T[0]
            self.gs_orbital  =   orbitals_T[0] + 1
            self.gs_spin     =   'Triplet'
            self.gs_ss       =   spins_T[0][0]
            self.gs_s        =   spins_T[0][1]

        # Restore all excited states
        if self.gs_spin == 'Singlet':
            self.singlet_energy  = [(self.gs_energy-x)*27.2114 for x in energies_S[1:]]
            self.singlet_orbital = [x + 1 for x in orbitals_S[1:]]
            self.singlet_spin    = spins_S[1:]
            self.singlet_sim     = sims_S[1:]
            self.triplet_energy  = [(self.gs_energy-x)*27.2114 for x in energies_T]
            self.triplet_orbital = [x + 1 for x in orbitals_T]
            self.triplet_spin    = spins_T
        else:
            self.singlet_energy  = [(self.gs_energy-x)*27.2114 for x in energies_S]
            self.singlet_orbital = [x + 1 for x in orbitals_S]
            self.singlet_spin    = spins_S
            self.singlet_sim     = sims_S
            self.triplet_energy  = [(self.gs_energy-x)*27.2114 for x in energies_T[1:]]
            self.triplet_orbital = [x + 1 for x in orbitals_T[1:]]
            self.triplet_spin    = spins_T[1:]
        
        # update the ground state energy
        # Note that the ground state energy is defined as the energy of the (N) system
        self.gs_energy = mf.e_tot - self.gs_energy
        
        # Print the excitation energies
        if mf.verbose >= logger.QUIET:
            log = logger.new_logger(mf, mf.verbose)
            log.log('###############################################################')
            log.log('##########         QEDFT Excited Energies          ############')
            log.log('###############################################################')
            log.log('Singlet excited energies (in eV):')
            log.log(", ".join([str(x) for x in self.singlet_energy]))
            log.log('Triplet excited energies (in eV):')
            log.log(", ".join([str(x) for x in self.triplet_energy]))
            log.log('')

        return self.singlet_energy, self.triplet_energy
    
    def analyze(self, verbose=None):
        """
        Placeholder for the analyze function.
        """
        # Print all information of excited states
        if verbose is None:
            verbose = self.mf.verbose
        log = logger.new_logger(self.mf, verbose)
        if log.verbose >= logger.NOTE:
            log.note('###############################################################')
            log.note('##############         QEDFT Analysis          ################')
            log.note('###############################################################')
            log.note('')
            log.note('The (N) system has a %s ground state', self.gs_spin)
            log.note('The ground state has spin <S^2>=%.6f, and 2S+1=%.6f', self.gs_ss, self.gs_s)
            log.note('')

        if log.verbose >= logger.NOTE:
            log.note('Singlet excited states:')
            for i in range(len(self.singlet_energy)):
                log.note('State %d: E = %.6f eV, <S^2> = %.6f, 2S+1 = %.6f',
                        i+1, self.singlet_energy[i],
                        self.singlet_spin[i][0], self.singlet_spin[i][1]) 
                if self.gs_spin == 'Singlet':
                    log.note('      %da --> %da', self.singlet_orbital[i], self.gs_orbital)
                else:
                    log.note('      %da --> %db', self.singlet_orbital[i], self.gs_orbital)
                log.note('      Orbital similarity: (%da, %.6f)', 
                        self.singlet_sim[i][0], self.singlet_sim[i][1])
            log.note('')    
            log.note('Triplet excited states:')
            for i in range(len(self.triplet_energy)):
                log.note('State %d: E = %.6f eV, <S^2> = %.6f, 2S+1 = %.6f',
                        i+1, self.triplet_energy[i],
                        self.triplet_spin[i][0], self.triplet_spin[i][1]) 
                if self.gs_spin == 'Singlet':
                    log.note('      %db --> %da', self.triplet_orbital[i], self.gs_orbital)
                else:
                    log.note('      %db --> %db', self.triplet_orbital[i], self.gs_orbital)
            log.note('')
        return None
    
    def vertical_ST_gap(self):
        """
        Calculate the singlet-triplet gap.

        Returns:
        -------
        gap_type : str
            The type of the singlet-triplet gap.
        E_ST : float
            The singlet-triplet gap in eV.
            If the ground state is singlet, it is S1-T1 gap;
            If the ground state is triplet, it is S1-T0 gap.
        """

        if self.gs_spin == 'Singlet':
            # If the ground state is singlet, we calculate the S1-T1 gap:
            return 'S1-T1', self.singlet_energy[0] - self.triplet_energy[0]
        else:
            # If the ground state is triplet, we calculate the S1-T0 gap:
            return 'S1-T0', self.singlet_energy[0]



QEDFT = QEDFT_Nmius1 = QEDFT_Doublet_Base
QEDFT_Nplus1 = QEDFT_Doublet_Base_Nplus1