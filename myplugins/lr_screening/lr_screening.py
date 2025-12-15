import numpy
from pyscf import lib
from pyscf import scf
from pyscf import tdscf
from pyscf import symm
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.scf import uhf_symm
from pyscf.tdscf import rhf
from pyscf.data import nist
from pyscf import __config__

OUTPUT_THRESHOLD = getattr(__config__, 'tdscf_rhf_get_nto_threshold', 0.3)
REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold', 1e-4)
MO_BASE = getattr(__config__, 'MO_BASE', 1)



def get_full_K(mf, mo_coeff=None, mo_occ=None):
    '''Get the full coupling matrix K.

    Args:
        mf : an instance of SCF class (e.g., RHF, UHF)
            The mean-field object from which the K matrix is derived.
        mo_coeff : ndarray
            Molecular orbital coefficients. If None, use mf.mo_coeff.

    Returns:
        K : ndarray
            The full coupling matrix K.
            K = (K_aaaa, K_aabb, K_bbbb), K_bbaa = K_aabb.transpose(2,3,0,1).
            For each block, in shape (nmo, nmo, nmo, nmo), 
            where nmo is the number of molecular orbitals.
    '''
    if isinstance(mf, scf.hf.RHF):
        mf = mf.to_uks() # must use to_uks(), NOT to_uhf().
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    
    mol = mf.mol
    nao = mol.nao_nr()
    mo_a = mo_coeff[0]
    mo_b = mo_coeff[1]
    nmo_a = mo_a.shape[1]
    nmo_b = mo_b.shape[1]

    k_aa = numpy.zeros((nmo_a, nmo_a, nmo_a, nmo_a))
    k_ab = numpy.zeros((nmo_a, nmo_a, nmo_b, nmo_b))
    k_bb = numpy.zeros((nmo_b, nmo_b, nmo_b, nmo_b))
    k = (k_aa, k_ab, k_bb)

    def add_hf_(k, hyb=1):
        eri_aa = ao2mo.general(mol, [mo_a,mo_a,mo_a,mo_a], compact=False)
        eri_ab = ao2mo.general(mol, [mo_a,mo_a,mo_b,mo_b], compact=False)
        eri_bb = ao2mo.general(mol, [mo_b,mo_b,mo_b,mo_b], compact=False)
        eri_aa = eri_aa.reshape(nmo_a,nmo_a,nmo_a,nmo_a)
        eri_ab = eri_ab.reshape(nmo_a,nmo_a,nmo_b,nmo_b)
        eri_bb = eri_bb.reshape(nmo_b,nmo_b,nmo_b,nmo_b)
        k_aa, k_ab, k_bb = k

        k_aa += numpy.einsum('pqsr->pqrs', eri_aa)
        k_aa -= numpy.einsum('prsq->pqrs', eri_aa) * hyb

        k_bb += numpy.einsum('pqsr->pqrs', eri_bb)
        k_bb -= numpy.einsum('prsq->pqrs', eri_bb) * hyb

        k_ab += numpy.einsum('pqsr->pqrs', eri_ab)
        return None
    
    if isinstance(mf, scf.hf.KohnShamDFT):
        from pyscf.dft import xc_deriv
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if mf.do_nlc():
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'derivative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        add_hf_(k, hyb)
        if omega != 0:  # For RSH
            with mol.with_range_coulomb(omega):
                eri_aa = ao2mo.general(mol, [mo_a,mo_a,mo_a,mo_a], compact=False)
                eri_bb = ao2mo.general(mol, [mo_b,mo_b,mo_b,mo_b], compact=False)
                eri_aa = eri_aa.reshape(nmo_a,nmo_a,nmo_a,nmo_a)
                eri_bb = eri_bb.reshape(nmo_b,nmo_b,nmo_b,nmo_b)
                k_aa, k_ab, k_bb = k
                fac = alpha - hyb
                k_aa -= numpy.einsum('prsq->pqrs', eri_aa) * fac
                k_bb -= numpy.einsum('prsq->pqrs', eri_bb) * fac
        
        xctype = ni._xc_type(mf.xc)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1, with_lapl=False)[0]
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc[:,0,:,0] * weight

                rho_a = lib.einsum('rp,pq->rq', ao, mo_a)
                rho_b = lib.einsum('rp,pq->rq', ao, mo_b)
                rho_pq_a = numpy.einsum('rp,rq->rpq', rho_a, rho_a)
                rho_pq_b = numpy.einsum('rp,rq->rpq', rho_b, rho_b)

                w_pq = numpy.einsum('rpq,r->rpq', rho_pq_a, wfxc[0,0])
                pqst = lib.einsum('rpq,rst->pqst', rho_pq_a, w_pq)
                k_aa += pqst

                # For the remaining part, I have not modified the index from iajb to pqst
                w_pq = numpy.einsum('ria,r->ria', rho_pq_b, wfxc[0,1])
                iajb = lib.einsum('ria,rjb->iajb', rho_pq_a, w_pq)
                k_ab += iajb

                w_pq = numpy.einsum('ria,r->ria', rho_pq_b, wfxc[1,1])
                iajb = lib.einsum('ria,rjb->iajb', rho_pq_b, w_pq)
                k_bb += iajb
        
        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight

                # Not optimized
                rho_o_a = lib.einsum('xrp,pi->xri', ao, mo_a)
                rho_v_a = lib.einsum('xrp,pi->xri', ao, mo_a)
                rho_o_b = lib.einsum('xrp,pi->xri', ao, mo_b)
                rho_v_b = lib.einsum('xrp,pi->xri', ao, mo_b)
                rho_ov_a = numpy.einsum('xri,ra->xria', rho_o_a, rho_v_a[0])
                rho_ov_b = numpy.einsum('xri,ra->xria', rho_o_b, rho_v_b[0])
                rho_ov_a[1:4] += numpy.einsum('ri,xra->xria', rho_o_a[0], rho_v_a[1:4])
                rho_ov_b[1:4] += numpy.einsum('ri,xra->xria', rho_o_b[0], rho_v_b[1:4])
                w_ov_aa = numpy.einsum('xyr,xria->yria', wfxc[0,:,0], rho_ov_a)
                w_ov_ab = numpy.einsum('xyr,xria->yria', wfxc[0,:,1], rho_ov_a)
                w_ov_bb = numpy.einsum('xyr,xria->yria', wfxc[1,:,1], rho_ov_b)

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_aa, rho_ov_a)
                k_aa += iajb

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_bb, rho_ov_b)
                k_bb += iajb            

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_ab, rho_ov_b)
                k_ab += iajb

        elif xctype == 'HF':
            pass

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')
        
        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight
                rho_oa = lib.einsum('xrp,pi->xri', ao, mo_a)
                rho_ob = lib.einsum('xrp,pi->xri', ao, mo_b)
                rho_va = lib.einsum('xrp,pi->xri', ao, mo_a)
                rho_vb = lib.einsum('xrp,pi->xri', ao, mo_b)
                rho_ov_a = numpy.einsum('xri,ra->xria', rho_oa, rho_va[0])
                rho_ov_b = numpy.einsum('xri,ra->xria', rho_ob, rho_vb[0])
                rho_ov_a[1:4] += numpy.einsum('ri,xra->xria', rho_oa[0], rho_va[1:4])
                rho_ov_b[1:4] += numpy.einsum('ri,xra->xria', rho_ob[0], rho_vb[1:4])
                tau_ov_a = numpy.einsum('xri,xra->ria', rho_oa[1:4], rho_va[1:4]) * .5
                tau_ov_b = numpy.einsum('xri,xra->ria', rho_ob[1:4], rho_vb[1:4]) * .5
                rho_ov_a = numpy.vstack([rho_ov_a, tau_ov_a[numpy.newaxis]])
                rho_ov_b = numpy.vstack([rho_ov_b, tau_ov_b[numpy.newaxis]])
                w_ov_aa = numpy.einsum('xyr,xria->yria', wfxc[0,:,0], rho_ov_a)
                w_ov_ab = numpy.einsum('xyr,xria->yria', wfxc[0,:,1], rho_ov_a)
                w_ov_bb = numpy.einsum('xyr,xria->yria', wfxc[1,:,1], rho_ov_b)

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_aa, rho_ov_a)
                k_aa += iajb

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_bb, rho_ov_b)
                k_bb += iajb

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_ab, rho_ov_b)
                k_ab += iajb

    else:
        add_hf_(k)

    return k


def get_full_eta(mf, fractional_spin=False):
    if isinstance(mf, scf.hf.RHF):
        mf = mf.to_uks() # must use to_uks(), NOT to_uhf().
        
    A, B = tdscf.uhf.get_ab(mf)
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

    M_full = (numpy.block([[A_aaaa, A_aabb], [A_bbaa, A_bbbb]])
              + numpy.block([[B_aaaa, B_aabb], [B_bbaa, B_bbbb]]))
    M_inv = numpy.linalg.pinv(M_full)

    del A_aaaa, A_aabb, A_bbbb, A_bbaa
    del B_aaaa, B_aabb, B_bbbb, B_bbaa
    del M_full

    M_inv_aaaa = M_inv[:Nocc_a*Nvir_a, :Nocc_a*Nvir_a].reshape(Nocc_a, Nvir_a, Nocc_a, Nvir_a)
    M_inv_bbbb = M_inv[Nocc_a*Nvir_a:, Nocc_a*Nvir_a:].reshape(Nocc_b, Nvir_b, Nocc_b, Nvir_b)
    M_inv_aabb = M_inv[:Nocc_a*Nvir_a, Nocc_a*Nvir_a:].reshape(Nocc_a, Nvir_a, Nocc_b, Nvir_b)
    M_inv_bbaa = M_inv[Nocc_a*Nvir_a:, :Nocc_a*Nvir_a].reshape(Nocc_b, Nvir_b, Nocc_a, Nvir_a)

    del M_inv

    k_aaaa, k_aabb, k_bbbb = get_full_K(mf)
    k_bbaa = k_aabb.transpose(2,3,0,1)

    eta_aa = (k_aaaa
             - lib.einsum('pqia,iajb,jbrs->pqrs', k_aaaa[:,:,:Nocc_a,Nocc_a:], M_inv_aaaa, k_aaaa[:Nocc_a,Nocc_a:,:,:])
             - lib.einsum('pqia,iajb,jbrs->pqrs', k_aabb[:,:,:Nocc_b,Nocc_b:], M_inv_bbaa, k_aaaa[:Nocc_a,Nocc_a:,:,:])
             - lib.einsum('pqia,iajb,jbrs->pqrs', k_aabb[:,:,:Nocc_b,Nocc_b:], M_inv_bbbb, k_bbaa[:Nocc_b,Nocc_b:,:,:])
             - lib.einsum('pqia,iajb,jbrs->pqrs', k_aaaa[:,:,:Nocc_a,Nocc_a:], M_inv_aabb, k_bbaa[:Nocc_b,Nocc_b:,:,:])
             - lib.einsum('pqai,iajb,bjrs->pqrs', k_aaaa[:,:,Nocc_a:,:Nocc_a], M_inv_aaaa, k_aaaa[Nocc_a:,:Nocc_a,:,:])
             - lib.einsum('pqai,iajb,bjrs->pqrs', k_aabb[:,:,Nocc_b:,:Nocc_b], M_inv_bbaa, k_aaaa[Nocc_a:,:Nocc_a,:,:])
             - lib.einsum('pqai,iajb,bjrs->pqrs', k_aabb[:,:,Nocc_b:,:Nocc_b], M_inv_bbbb, k_bbaa[Nocc_b:,:Nocc_b,:,:])
             - lib.einsum('pqai,iajb,bjrs->pqrs', k_aaaa[:,:,Nocc_a:,:Nocc_a], M_inv_aabb, k_bbaa[Nocc_b:,:Nocc_b,:,:])
             )
    eta_bb = (k_bbbb
             - lib.einsum('pqia,iajb,jbrs->pqrs', k_bbaa[:,:,:Nocc_a,Nocc_a:], M_inv_aaaa, k_aabb[:Nocc_a,Nocc_a:,:,:])
             - lib.einsum('pqia,iajb,jbrs->pqrs', k_bbbb[:,:,:Nocc_b,Nocc_b:], M_inv_bbaa, k_aabb[:Nocc_a,Nocc_a:,:,:])
             - lib.einsum('pqia,iajb,jbrs->pqrs', k_bbbb[:,:,:Nocc_b,Nocc_b:], M_inv_bbbb, k_bbbb[:Nocc_b,Nocc_b:,:,:])
             - lib.einsum('pqia,iajb,jbrs->pqrs', k_bbaa[:,:,:Nocc_a,Nocc_a:], M_inv_aabb, k_bbbb[:Nocc_b,Nocc_b:,:,:])
             - lib.einsum('pqai,iajb,bjrs->pqrs', k_bbaa[:,:,Nocc_a:,:Nocc_a], M_inv_aaaa, k_aabb[Nocc_a:,:Nocc_a,:,:])
             - lib.einsum('pqai,iajb,bjrs->pqrs', k_bbbb[:,:,Nocc_b:,:Nocc_b], M_inv_bbaa, k_aabb[Nocc_a:,:Nocc_a,:,:])
             - lib.einsum('pqai,iajb,bjrs->pqrs', k_bbbb[:,:,Nocc_b:,:Nocc_b], M_inv_bbbb, k_bbbb[Nocc_b:,:Nocc_b,:,:])
             - lib.einsum('pqai,iajb,bjrs->pqrs', k_bbaa[:,:,Nocc_a:,:Nocc_a], M_inv_aabb, k_bbbb[Nocc_b:,:Nocc_b,:,:])
             )
    if not fractional_spin:
        return (numpy.real(eta_aa), numpy.real(eta_bb))
    else:
        eta_ab = ( k_aabb
                 - lib.einsum('pqia,iajb,jbrs->pqrs', k_aaaa[:,:,:Nocc_a,Nocc_a:], M_inv_aaaa, k_aabb[:Nocc_a,Nocc_a:,:,:])
                 - lib.einsum('pqia,iajb,jbrs->pqrs', k_aabb[:,:,:Nocc_b,Nocc_b:], M_inv_bbaa, k_aabb[:Nocc_a,Nocc_a:,:,:])
                 - lib.einsum('pqia,iajb,jbrs->pqrs', k_aabb[:,:,:Nocc_b,Nocc_b:], M_inv_bbbb, k_bbbb[:Nocc_b,Nocc_b:,:,:])
                 - lib.einsum('pqia,iajb,jbrs->pqrs', k_aaaa[:,:,:Nocc_a,Nocc_a:], M_inv_aabb, k_bbbb[:Nocc_b,Nocc_b:,:,:])
                 - lib.einsum('pqai,iajb,bjrs->pqrs', k_aaaa[:,:,Nocc_a:,:Nocc_a], M_inv_aaaa, k_aabb[Nocc_a:,:Nocc_a,:,:])
                 - lib.einsum('pqai,iajb,bjrs->pqrs', k_aabb[:,:,Nocc_b:,:Nocc_b], M_inv_bbaa, k_aabb[Nocc_a:,:Nocc_a,:,:])
                 - lib.einsum('pqai,iajb,bjrs->pqrs', k_aabb[:,:,Nocc_b:,:Nocc_b], M_inv_bbbb, k_bbbb[Nocc_b:,:Nocc_b,:,:])
                 - lib.einsum('pqai,iajb,bjrs->pqrs', k_aaaa[:,:,Nocc_a:,:Nocc_a], M_inv_aabb, k_bbbb[Nocc_b:,:Nocc_b,:,:])
                 )
        return (numpy.real(eta_aa), numpy.real(eta_bb), numpy.real(eta_ab))


def get_full_kappa(mf, u, fractional_spin=False):   
    u_a, u_b = u
    if not fractional_spin:
        eta_a, eta_b = get_full_eta(mf, fractional_spin=fractional_spin)
        # for i1 in range(2):
        #     for i2 in range(2):
        #         for i3 in range(2):
        #             for i4 in range(2):
        #                 print('eta_a_{}{}{}{}'.format(i1,i2,i3,i4), eta_a[i1,i2,i3,i4])
        kappa_a = lib.einsum('ps,pt,stuv,qu,qv->pq', u_a, u_a, eta_a, u_a, u_a)
        kappa_b = lib.einsum('ps,pt,stuv,qu,qv->pq', u_b, u_b, eta_b, u_b, u_b)
        return (numpy.real(kappa_a), numpy.real(kappa_b))
    else:
        eta_a, eta_b, eta_ab = get_full_eta(mf, fractional_spin=fractional_spin)
        kappa_a = lib.einsum('ps,pt,stuv,qu,qv->pq', u_a, u_a, eta_a, u_a, u_a)
        kappa_b = lib.einsum('ps,pt,stuv,qu,qv->pq', u_b, u_b, eta_b, u_b, u_b)
        kappa_ab = lib.einsum('ps,pt,stuv,qu,qv->pq', u_a, u_a, eta_ab, u_b, u_b)
    return (numpy.real(kappa_a), numpy.real(kappa_b), numpy.real(kappa_ab))
    