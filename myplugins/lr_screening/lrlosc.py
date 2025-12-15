import numpy
from pyscf import scf, tdscf, lib
from myplugins.lr_screening import lr_screening

def FS_Lambda(lbd_a, lbd_b):
    # 提取对角线元素
    diag_a = numpy.diag(lbd_a)
    diag_b = numpy.diag(lbd_b)
    
    # 使用广播创建网格
    a_grid = diag_a[:, numpy.newaxis]  # 变为列向量
    b_grid = diag_b[numpy.newaxis, :]  # 变为行向量
    
    # 计算条件和结果
    condition = a_grid + b_grid > 1
    result = numpy.where(condition, 
                     (1 - a_grid) * (1 - b_grid), 
                     a_grid * b_grid)
    
    return result

def RKS_correction(mf, losc_data, fractional_spin=False):
    mo_coeff = mf.mo_coeff
    assert (mo_coeff.dtype == numpy.double)
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)


    select_CO_idx   =   losc_data['select_CO_idx']
    U               =   losc_data['U']
    select_CO       =   select_CO_idx[0]

    # (1) Prepare the U matrices and kappa matrices
    Umat             =   numpy.eye(nao, dtype=float) # in shape(nao,nao)
    if select_CO is not None:
        Umat[select_CO[0]:select_CO[1], select_CO[0]:select_CO[1]] = 1 * U[0]

    Upo        =   Umat[:,:nocc] # in shape(nao,nocc)
    Upv        =   Umat[:,nocc:] # in shape(nao,nvir)

    lbd = numpy.real(lib.einsum('pi,qi->pq', Upo, Upo)) # local occupation number

    if not fractional_spin:
        kappa, _ = lr_screening.get_full_kappa(mf, (Umat,Umat), fractional_spin=fractional_spin)
    else:
        kappa, _, kappa_ab = lr_screening.get_full_kappa(mf, (Umat,Umat), fractional_spin=fractional_spin)

    # (2) Prepare the orbital energy corrections
    identical = numpy.eye(nao, dtype=float)
    d_eigs = lib.einsum('pq,pq,ps,qs->s', kappa, identical-2*lbd, Umat, Umat) * 0.5

    # (3) Prepare the total energy corrections
    d_etot = 2 * lib.einsum('pq,pq,pq->', kappa, identical-lbd, lbd)
    if fractional_spin:
        FS_lbd = FS_Lambda(lbd, lbd)
        d_etot -= lib.einsum('pq,pq->', kappa_ab, FS_lbd)

    # (4) Update losc_data
    losc_data['losc_energy']        = d_etot
    losc_data['losc_dfa_energy']    = mf.e_tot + d_etot
    losc_data['losc_dfa_orbital_energy'] = mf.mo_energy + d_eigs
    return (mf.e_tot + d_etot), sorted(mf.mo_energy + d_eigs)


def UKS_correction(mf, losc_data, fractional_spin=False):
    mo_coeff = mf.mo_coeff
    assert (mo_coeff[0].dtype == numpy.double)
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff[0].shape
    occidx_a = numpy.where(mo_occ[0]>0)[0]
    viridx_a = numpy.where(mo_occ[0]==0)[0]
    occidx_b = numpy.where(mo_occ[1]>0)[0]
    viridx_b = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidx_a)
    nvira = len(viridx_a)
    noccb = len(occidx_b)
    nvirb = len(viridx_b)

    select_CO_idx   =   losc_data['select_CO_idx']
    U               =   losc_data['U']
    select_CO_a     =   select_CO_idx[0]
    select_CO_b     =   select_CO_idx[1]

    # (1) Prepare the U matrices and kappa matrices
    U_a             =   numpy.eye(nao, dtype=float) # in shape(nao,nao)
    U_b             =   numpy.eye(nao, dtype=float) # in shape(nao,nao)
    if select_CO_a is not None:
        U_a[select_CO_a[0]:select_CO_a[1], select_CO_a[0]:select_CO_a[1]] = 1 * U[0]
    if select_CO_b is not None:
        U_b[select_CO_b[0]:select_CO_b[1], select_CO_b[0]:select_CO_b[1]] = 1 * U[1]
    Upo_a        =   U_a[:,:nocca] # in shape(nao,nocca)
    Upv_a        =   U_a[:,nocca:] # in shape(nao,nvira)
    Upo_b        =   U_b[:,:noccb] # in shape(nao,noccb)
    Upv_b        =   U_b[:,noccb:] # in shape(nao,nvirb)

    lbd_a = numpy.real(lib.einsum('pi,qi->pq', Upo_a, Upo_a)) # local occupation number
    lbd_b = numpy.real(lib.einsum('pi,qi->pq', Upo_b, Upo_b)) # local occupation number

    if not fractional_spin:
        kappa_a, kappa_b = lr_screening.get_full_kappa(mf, (U_a,U_b), fractional_spin=fractional_spin)
    else:
        kappa_a, kappa_b, kappa_ab = lr_screening.get_full_kappa(mf, (U_a,U_b), fractional_spin=fractional_spin)

    # (2) Prepare the orbital energy corrections
    identical = numpy.eye(nao, dtype=float)
    d_eigs_a = lib.einsum('pq,pq,ps,qs->s', kappa_a, identical-2*lbd_a, U_a, U_a) * 0.5
    d_eigs_b = lib.einsum('pq,pq,ps,qs->s', kappa_b, identical-2*lbd_b, U_b, U_b) * 0.5


    # (3) Prepare the total energy corrections
    d_etot = lib.einsum('pq,pq,pq->', kappa_a, identical-lbd_a, lbd_a)
    print(kappa_a[select_CO_a[0]:select_CO_a[1], select_CO_a[0]:select_CO_a[1]])
    print(U[0])
    d_etot += lib.einsum('pq,pq,pq->', kappa_b, identical-lbd_b, lbd_b)
    if fractional_spin:
        FS_lbd = FS_Lambda(lbd_a, lbd_b)
        d_etot -= lib.einsum('pq,pq->', kappa_ab, FS_lbd)

    # (4) Update losc_data
    losc_data['losc_energy']        = d_etot
    losc_data['losc_dfa_energy']    = mf.e_tot + d_etot
    losc_data['losc_dfa_orbital_energy'] = (mf.mo_energy[0] + d_eigs_a, mf.mo_energy[1] + d_eigs_b)
    return (mf.e_tot + d_etot), (sorted(mf.mo_energy[0] + d_eigs_a), sorted(mf.mo_energy[1] + d_eigs_b))