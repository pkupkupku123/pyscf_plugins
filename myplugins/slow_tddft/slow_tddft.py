import numpy
from pyscf import scf, tdscf, lib
from pyscf.lib import logger
from pyscf.data import nist

def energy_filter(e, x1, energy_window=None):
    """
    Filter eigenvalues and eigenvectors by energy window, transpose eigenvectors,
    and sort by energy in ascending order.
    """
    e_ev = numpy.real(e * 27.2114)  # convert to eV and take real part

    # find the mask for the energy window
    if energy_window is None:
        mask = slice(None)  # select all
    elif energy_window[0] is None and energy_window[1] is None:
        mask = slice(None)
    elif energy_window[0] is None:
        mask = (e_ev <= energy_window[1])
    elif energy_window[1] is None:
        mask = (e_ev >= energy_window[0])
    else:
        if energy_window[0] >= energy_window[1]:
            raise ValueError("Energy window is invalid: %s" % str(energy_window))
        mask = (e_ev >= energy_window[0]) & (e_ev <= energy_window[1])
    
    # if there's no eigenvalue in the energy window, raise an error
    if not mask.any():
        raise ValueError("No eigenvalue in the energy window: %s" % str(energy_window))

    # find the indices of the eigenvalues that are within the energy window
    e_filtered = numpy.real(e[mask])
    x1_filtered = x1.T[mask]  # transpose the eigenvectors

    # sort the eigenvalues and eigenvectors by energy
    sorted_indices = numpy.argsort(e_filtered)
    e_sorted = e_filtered[sorted_indices]
    x1_sorted = x1_filtered[sorted_indices]

    return e_sorted, x1_sorted
    

#region slow_TDA_R
class slow_TDA_R(tdscf.rhf.TDBase):
    """
    Class for sloving RTDA equations by directly diagonalizing the orbital rotation Hessian.
    """
    energy_window = [None, None]

    def get_orhess(self):
        """
        Get the orbital rotation Hessian.
        """
        if self.singlet:
            A, _ = tdscf.rhf.get_ab(self._scf)
            Nocc, Nvir = A.shape[0], A.shape[1]
            return A.reshape(Nocc * Nvir, Nocc * Nvir)
        else:
            raise ValueError("Triplet excitation not implemented for slow RTDA.")


    def kernel(self):
        self.e, x1 = numpy.linalg.eigh(self.get_orhess())
        self.e, x1 = energy_filter(self.e, x1, self.energy_window)
        # After the filter, x1 is transposed
        self.converged = [True for e in self.e]

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        # 1/sqrt(2) because self.x is for alpha excitation and 2(X^+*X) = 1
        self.xy = [(xi.reshape(nocc,nvir)*numpy.sqrt(.5),0) for xi in x1]
        self._finalize()
        return self.e, self.xy, x1
    

#region slow_TDA_U
class slow_TDA_U(tdscf.uhf.TDBase):
    """
    Class for sloving UTDA equations by directly diagonalizing the orbital rotation Hessian.
    """
    energy_window = [None, None]

    def __init__(self, mf):
        if isinstance(mf, scf.hf.RHF):
            mf = mf.to_uks() # must use to_uks(), NOT to_uhf().
        super().__init__(mf)
        

    def get_orhess(self):
        """
        Get the orbital rotation Hessian.
        """
        A, _ = tdscf.uhf.get_ab(self._scf)
        A_aaaa, A_aabb, A_bbbb = A
        A_bbaa = A_aabb.transpose(2,3,0,1)
        Nocc_a, Nvir_a = A_aaaa.shape[0], A_aaaa.shape[1]
        Nocc_b, Nvir_b = A_bbbb.shape[0], A_bbbb.shape[1]
        A_aaaa = A_aaaa.reshape(Nocc_a * Nvir_a, Nocc_a * Nvir_a)
        A_bbbb = A_bbbb.reshape(Nocc_b * Nvir_b, Nocc_b * Nvir_b)
        A_aabb = A_aabb.reshape(Nocc_a * Nvir_a, Nocc_b * Nvir_b)
        A_bbaa = A_bbaa.reshape(Nocc_b * Nvir_b, Nocc_a * Nvir_a)
        return numpy.block([[A_aaaa, A_aabb], [A_bbaa, A_bbbb]])
    

    def kernel(self):
        self.e, x1 = numpy.linalg.eigh(self.get_orhess())
        self.e, x1 = energy_filter(self.e, x1, self.energy_window)
        self.converged = [True for e in self.e]
        nmo = self._scf.mo_occ[0].size
        nocca = (self._scf.mo_occ[0]>0).sum()
        noccb = (self._scf.mo_occ[1]>0).sum()
        nvira = nmo - nocca
        nvirb = nmo - noccb
        self.xy = [((xi[:nocca*nvira].reshape(nocca,nvira),  # X_alpha
                    xi[nocca*nvira:].reshape(noccb,nvirb)), # X_beta
                    (0, 0))  # (Y_alpha, Y_beta)
                    for xi in x1]
        self._finalize()
        return self.e, self.xy, x1
    

#region slow_TDA
def slow_TDA(mf):
    """
    Function to slove TDA equations by directly diagonalizing the orbital rotation Hessian.
    """
    if isinstance(mf, scf.hf.RHF):
        return slow_TDA_R(mf)
    elif isinstance(mf, scf.uhf.UHF):
        return slow_TDA_U(mf)
    else:
        raise ValueError("Unsupported SCF type: %s" % type(mf))
    

#region slow_TDDFT_R
class slow_TDDFT_R(tdscf.rhf.TDBase):
    """
    Class for sloving Restricted TDDFT equations 
    by directly diagonalizing the orbital rotation Hessian.
    """
    energy_window = [0, None]

    def get_orhess(self):
        """
        Get the orbital rotation Hessian.
        """
        if self.singlet:
            A, B = tdscf.rhf.get_ab(self._scf)
            Nocc, Nvir = A.shape[0], A.shape[1]
            A = A.reshape(Nocc * Nvir, Nocc * Nvir)
            B = B.reshape(Nocc * Nvir, Nocc * Nvir)
            # Currently, only real-valued orbitals are supported
            #  [ A ,  B ]
            #  [-B*, -A*] in principle
            return numpy.block([[A, B], [-B, -A]])
        else:
            raise ValueError("Triplet excitation not implemented for slow RTDA.")
        
    def kernel(self):
        self.e, x1 = numpy.linalg.eig(self.get_orhess())
        self.e, x1 = energy_filter(self.e, x1, self.energy_window)
        # After the filter, x1 is transposed
        self.converged = [True for e in self.e]

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        def norm_xy(z):
            x, y = z.reshape(2,nocc,nvir)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            norm = numpy.sqrt(.5/norm)  # normalize to 0.5 for alpha spin
            return x*norm, y*norm
        self.xy = [norm_xy(z) for z in x1]
        self._finalize()
        return self.e, self.xy
    

#region slow_TDDFT_U
class slow_TDDFT_U(tdscf.uhf.TDBase):
    """
    Class for sloving Unrestricted TDDFT equations 
    by directly diagonalizing the orbital rotation Hessian.
    """
    energy_window = [0, None]

    def __init__(self, mf):
        if isinstance(mf, scf.hf.RHF):
            mf = mf.to_uks() # must use to_uks(), NOT to_uhf().
        super().__init__(mf)
        

    def get_orhess(self):
        """
        Get the orbital rotation Hessian.
        """
        A, B = tdscf.uhf.get_ab(self._scf)
        A_aaaa, A_aabb, A_bbbb = A
        B_aaaa, B_aabb, B_bbbb = B
        del A, B

        A_bbaa = A_aabb.transpose(2,3,0,1)
        B_bbaa = B_aabb.transpose(2,3,0,1)
        Nocc_a, Nvir_a = A_aaaa.shape[0], A_aaaa.shape[1]
        Nocc_b, Nvir_b = A_bbbb.shape[0], A_bbbb.shape[1]
        A_aaaa = A_aaaa.reshape(Nocc_a * Nvir_a, Nocc_a * Nvir_a)
        A_bbbb = A_bbbb.reshape(Nocc_b * Nvir_b, Nocc_b * Nvir_b)
        A_aabb = A_aabb.reshape(Nocc_a * Nvir_a, Nocc_b * Nvir_b)
        A_bbaa = A_bbaa.reshape(Nocc_b * Nvir_b, Nocc_a * Nvir_a)
        B_aaaa = B_aaaa.reshape(Nocc_a * Nvir_a, Nocc_a * Nvir_a)
        B_bbbb = B_bbbb.reshape(Nocc_b * Nvir_b, Nocc_b * Nvir_b)
        B_aabb = B_aabb.reshape(Nocc_a * Nvir_a, Nocc_b * Nvir_b)
        B_bbaa = B_bbaa.reshape(Nocc_b * Nvir_b, Nocc_a * Nvir_a)
        # Currently, only real-valued orbitals are supported
        #  [ A ,  B ]
        #  [-B*, -A*] in principle
        A = numpy.block([[A_aaaa, A_aabb], [A_bbaa, A_bbbb]])
        B = numpy.block([[B_aaaa, B_aabb], [B_bbaa, B_bbbb]])
        return numpy.block([[A, B], [-B, -A]])
    
    def kernel(self):
        self.e, x1 = numpy.linalg.eig(self.get_orhess())
        self.e, x1 = energy_filter(self.e, x1, self.energy_window)
        self.converged = [True for e in self.e]
        nmo = self._scf.mo_occ[0].size
        nocca = (self._scf.mo_occ[0]>0).sum()
        noccb = (self._scf.mo_occ[1]>0).sum()
        nvira = nmo - nocca
        nvirb = nmo - noccb
        xy = []
        for i, z in enumerate(x1):
            x, y = z.reshape(2,-1)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            if norm > 0:
                norm = 1/numpy.sqrt(norm)
                xy.append(((x[:nocca*nvira].reshape(nocca,nvira) * norm,  # X_alpha
                            x[nocca*nvira:].reshape(noccb,nvirb) * norm), # X_beta
                           (y[:nocca*nvira].reshape(nocca,nvira) * norm,  # Y_alpha
                            y[nocca*nvira:].reshape(noccb,nvirb) * norm)))# Y_beta
        self.xy = xy
        self._finalize()
        return self.e, self.xy
    

#region slow_TDDFT
def slow_TDDFT(mf):
    """
    Function to slove TDDFT equations by directly diagonalizing the orbital rotation Hessian.
    """
    if isinstance(mf, scf.hf.RHF):
        return slow_TDDFT_R(mf)
    elif isinstance(mf, scf.uhf.UHF):
        return slow_TDDFT_U(mf)
    else:
        raise ValueError("Unsupported SCF type: %s" % type(mf))