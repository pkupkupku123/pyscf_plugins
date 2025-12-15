import numpy
from functools import reduce

from pyscf import lib, dft
from pyscf.lib import logger
from pyscf.soscf import ciah
from pyscf.lo import orth, cholesky_mos
from pyscf import __config__



def L_mat_1(mol, mo_coeff, charge_center=None):
    '''Compute the L matrix for dual localization.
    L_ij = \delta_ij <i|r^2|j> - <i|r|j>^2
    '''
    if charge_center is None:
        charge_center = (numpy.einsum('z,zx->x', mol.atom_charges(), mol.atom_coords())
                         / mol.atom_charges().sum())
    with mol.with_common_origin(charge_center):
        dip = numpy.asarray([reduce(lib.dot, (mo_coeff.conj().T, x, mo_coeff))
                             for x in mol.intor_symmetric('int1e_r', comp=3)])
        r2int = mol.intor_symmetric('int1e_r2')
        r2 = reduce(lib.dot, (mo_coeff.conj().T, r2int, mo_coeff))
    Lmat = (numpy.diag(numpy.diag(r2)) 
            - lib.einsum('xij,xij->ij', dip, dip))
    return Lmat


class Dual_Localizer():
    def __init__(self, mol, mo_energy, mo_coeff, gamma=0.5, c=20.0, Lversion=1):
        self.mol = mol
        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.gamma = gamma   
        self.c = c
        if Lversion == 1:
            self.L_mat = L_mat_1
        else:
            raise NotImplementedError("Only Lversion=1 is implemented.")

    def set_c(self, c):
        self.c = c

    def set_gamma(self, gamma):
        self.gamma = gamma

    def kernel(self):
        Hamiltonian = ((1-self.gamma) * self.L_mat(self.mol, self.mo_coeff) 
                        + self.gamma * self.c * numpy.diag(self.mo_energy))
        _, eigc = numpy.linalg.eigh(Hamiltonian)
        C_lo = numpy.dot(self.mo_coeff, eigc)
        U = eigc.T
        return C_lo, U
    
    def kernel_test(self):
        Hamiltonian = ((1-self.gamma) * self.L_mat(self.mol, self.mo_coeff) 
                        + self.gamma * self.c * numpy.diag(self.mo_energy))
        print("Hermitian check:", numpy.allclose(Hamiltonian, Hamiltonian.conj().T))
        eigv, eigc = numpy.linalg.eigh(Hamiltonian)
        C_lo = numpy.dot(self.mo_coeff, eigc)
        U = eigc.T
        return C_lo, U, eigv



def test():
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = '''H  0.000000  0.000000  0.000000
                  H  0.000000  0.000000  5.20000'''
    mol.unit = 'A'
    mol.charge = 1
    mol.spin = 1
    mol.basis = 'def2-qzvppd'
    mol.build()

    mf = scf.UKS(mol)
    mf.xc = 'BLYP'
    mf.kernel()
    print(numpy.shape(mf.mo_coeff))

    localizer = Dual_Localizer(mol, mf.mo_energy[0][:15], mf.mo_coeff[0][:,:15]
                               , gamma=0.507, c=20.0)
    localizer.set_gamma(0.5)
    C_lo, U, eigv = localizer.kernel_test()
    print("Rotation matrix U:")
    print(U)
    print('Local occupation matrix:')
    print(numpy.einsum('pi,qi->pq', U[:,:1], U[:,:1]))
    # print('Localized MO coefficients:')
    # print(C_lo)
    print("Eigenvalues:")
    print(eigv)
    return None


if __name__ == '__main__':
    test()