import numpy as np
from pyscf import dft, gto


def aclda1(alpha, Cx):
    def eval_x_aclda(xc_code, rho, spin, relativity=0, deriv=1, omega=None, verbose=None):
        ex, vx, fx, kx = None, None, None, None
        if spin == 0:  # ===== 非自旋极化情况 =====
            n = rho
            n += 1e-14
            rho_aux = n ** (alpha * 3.0 / 4.0)
            ex_aux, vx_aux, fx_aux, _ = dft.libxc.eval_xc('LDA_X', rho_aux, spin, relativity, deriv, omega, verbose)
            factor = n ** (alpha * 3.0 / 4.0 - 1)
            ex = Cx * factor * ex_aux
            
            if deriv > 0:
                vrho_aux = vx_aux[0]
                vrho = Cx * alpha * 3.0 / 4.0 * factor * vrho_aux
                vx = (vrho,)
            if deriv > 1:
                v2rho2_aux = fx_aux[0]
                v2rho2 = Cx * (alpha * 3.0 / 4.0 * factor) ** 2 * v2rho2_aux + \
                         Cx * (3.0 * alpha / 4.0) * (3.0 * alpha / 4.0 - 1) * n ** (alpha * 3.0 / 4.0 - 2) * vrho_aux
                fx = (v2rho2,)

        else:  # ===== 自旋极化情况 =====
            rho_a, rho_b = rho[0], rho[1]
            n_a, n_b = rho_a, rho_b
            ex_a, vx_a, fx_a, _ = eval_x_aclda('LDA_X_aclda', 2*rho_a, spin=0, relativity=relativity, 
                                                    deriv=deriv, omega=omega, verbose=verbose)
            ex_b, vx_b, fx_b, _ = eval_x_aclda('GGA_X_acPBE', 2*rho_b, spin=0, relativity=relativity, 
                                                    deriv=deriv, omega=omega, verbose=verbose)
            n_total = n_a + n_b
            ex = np.zeros_like(n_total)
            for i in range(len(n_total)):
                ex[i] = n_a[i] / n_total[i] * ex_a[i] + n_b[i] / n_total[i] * ex_b[i]
            if deriv > 0:
                vrho_a, vrho_b = vx_a[0], vx_b[0]
                vrho = np.array([vrho_a, vrho_b]).T
                vx = (vrho,)
            if deriv > 1:
                v2rho2_a, v2rho2_b = fx_a[0], fx_b[0]
                v2rho2 = np.array([
                    2 * v2rho2_a,                # rho_a rho_a
                    np.zeros_like(n_total),  # rho_a rho_b
                    2 * v2rho2_b                 # rho_b rho_b
                    ]).T 
                fx = (v2rho2,)

        return ex, vx, fx, kx

    def eval_c_aclda(xc_code, rho, spin, relativity=0, deriv=1, omega=None, verbose=None):
        ec, vc, fc, _ = dft.libxc.eval_xc('LDA_C_VWN', rho, spin, relativity, deriv, omega, verbose)
        return ec, vc, fc, None


    def eval_xc_aclda(xc_code, rho, spin, relativity=0, deriv=1, omega=None, verbose=None):
        # ex, vx, fx, _ = dft.libxc.eval_xc('LDA_X', rho, spin, relativity, deriv, omega, verbose)
        # ec, vc, fc, _ = dft.libxc.eval_xc('LDA_C_VWN', rho, spin, relativity, deriv, omega, verbose)
        ex, vx, fx, _ = eval_x_aclda('LDA_X', rho, spin, relativity, deriv, omega, verbose)
        ec, vc, fc, _ = eval_c_aclda('LDA_C_VWN', rho, spin, relativity, deriv, omega, verbose)

        exc = ex + ec
        vxc, fxc = None, None
        if deriv > 0:
            vrho_x = vx[0] # type: ignore
            vrho_c = vc[0]
            vrho = vrho_x + vrho_c
            vxc = (vrho,)
        if deriv > 1:
            v2rho2_x = fx[0] # type: ignore
            v2rho2_c = fc[0]
            v2rho2 = v2rho2_x + v2rho2_c
            fxc = (v2rho2,)

        return exc, vxc, fxc, None

    return eval_xc_aclda


def aclda2(alpha, Cx):
    def eval_xc_aclda(xc_code, rho, spin, relativity=0, deriv=1, omega=None, verbose=None):
        exc, vxc, fxc, kxc = None, None, None, None
        if spin == 0:  # ===== 非自旋极化情况 =====
            n = rho
            n += 1e-14
            rho_aux = n ** (alpha * 3.0 / 4.0)
            exc_aux, vxc_aux, fxc_aux, _ = dft.libxc.eval_xc('LDA, VWN', rho_aux, spin, relativity, deriv, omega, verbose)
            factor = n ** (alpha * 3.0 / 4.0 - 1)
            exc = Cx * factor * exc_aux
            
            if deriv > 0:
                vrho_aux = vxc_aux[0]
                vrho = Cx * alpha * 3.0 / 4.0 * factor * vrho_aux
                vxc = (vrho,)
            if deriv > 1:
                v2rho2_aux = fxc_aux[0]
                v2rho2 = Cx * (alpha * 3.0 / 4.0 * factor) ** 2 * v2rho2_aux + \
                         Cx * (3.0 * alpha / 4.0) * (3.0 * alpha / 4.0 - 1) * n ** (alpha * 3.0 / 4.0 - 2) * vrho_aux
                fxc = (v2rho2,)

        else:  # ===== 自旋极化情况 =====
            rho_a, rho_b = rho[0], rho[1]
            n_a, n_b = rho_a + 1e-14, rho_b + 1e-14
            n_a_aux, n_b_aux = n_a ** (alpha * 3.0 / 4.0), n_b ** (alpha * 3.0 / 4.0)
            rho_a_aux = n_a_aux
            rho_b_aux = n_b_aux
            rho_aux = np.array([rho_a_aux, rho_b_aux])
            exc_aux, vxc_aux, fxc_aux, _ = dft.libxc.eval_xc('LDA, VWN', rho_aux, spin, relativity, deriv, omega, verbose)
            n_total_aux = n_a_aux + n_b_aux
            n_total = n_a + n_b
            exc = Cx * n_total_aux / n_total * exc_aux
            if deriv > 0:
                vrho_aux = vxc_aux[0]
                vrho_a_aux, vrho_b_aux = vrho_aux.T
                vrho_a = Cx * alpha * 3.0 / 4.0 * n_a ** (alpha * 3.0 / 4.0 - 1) * vrho_a_aux
                vrho_b = Cx * alpha * 3.0 / 4.0 * n_b ** (alpha * 3.0 / 4.0 - 1) * vrho_b_aux
                vrho = np.array([vrho_a, vrho_b]).T
                vxc = (vrho,)
            if deriv > 1:
                v2rho2_aux = fxc_aux[0]
                v2rho2_aa_aux, v2rho2_ab_aux, v2rho2_bb_aux = v2rho2_aux.T
                v2rho2_aa = Cx * (alpha * 3.0 / 4.0 * n_a ** (alpha * 3.0 / 4.0 - 1)) ** 2 * v2rho2_aa_aux + \
                            Cx * (3.0 * alpha / 4.0) * (3.0 * alpha / 4.0 - 1) * n_a ** (alpha * 3.0 / 4.0 - 2) * vrho_a_aux
                v2rho2_bb = Cx * (alpha * 3.0 / 4.0 * n_b ** (alpha * 3.0 / 4.0 - 1)) ** 2 * v2rho2_bb_aux + \
                            Cx * (3.0 * alpha / 4.0) * (3.0 * alpha / 4.0 - 1) * n_b ** (alpha * 3.0 / 4.0 - 2) * vrho_b_aux
                v2rho2_ab = Cx * (alpha * 3.0 / 4.0) ** 2 * n_a ** (alpha * 3.0 / 4.0 - 1) * n_b ** (alpha * 3.0 / 4.0 - 1) * v2rho2_ab_aux
                v2rho2 = np.array([
                    v2rho2_aa,
                    v2rho2_ab,
                    v2rho2_bb
                    ]).T 
                fxc = (v2rho2,)

        return exc, vxc, fxc, kxc

    return eval_xc_aclda


def acpbe1(alpha, Cx):
    def eval_x_acpbe1(xc_code, rho, spin, relativity=0, deriv=1, omega=None, verbose=None):
        ex, vx, fx, kx = None, None, None, None
        if spin == 0:  # ===== 非自旋极化情况 =====
            ex_aux, vx_aux, fx_aux, _ = dft.libxc.eval_xc('GGA_X_PBE', rho, spin, relativity, deriv, omega, verbose)
            n = rho[0]
            n += 1e-14
            factor = Cx * n ** (alpha - 4.0 / 3.0)
            ex = factor * ex_aux
            if deriv > 0:
                vrho_aux, vsigma_aux = vx_aux
                vrho = factor * (vrho_aux + (alpha - 4.0 / 3.0) * ex_aux)
                vsigma = factor * vsigma_aux
                vx = (vrho, vsigma)
            if deriv > 1:
                v2rho2_aux, v2rhosigma_aux, v2sigma2_aux = fx_aux
                v2rho2 = Cx * (alpha - 4.0 / 3.0) * (alpha - 7.0 / 3.0) * n ** (alpha - 7.0 / 3.0) * ex_aux + \
                         Cx * 2 * (alpha - 4.0 / 3.0) * n ** (alpha - 7.0 / 3.0) * vrho_aux + \
                         Cx * n ** (alpha - 4.0 / 3.0) * v2rho2_aux
                v2rhosigma = Cx * (alpha - 4.0 / 3.0) * n ** (alpha - 7.0 / 3.0) * vsigma_aux + \
                            Cx * n ** (alpha - 4.0 / 3.0) * v2rhosigma_aux
                v2sigma2 = Cx * n ** (alpha - 4.0 / 3.0) * v2sigma2_aux
                fx = (v2rho2, v2rhosigma, v2sigma2)
        else:  # ===== 自旋极化情况 =====
            rho_a, rho_b = rho[0], rho[1]
            n_a, n_b = rho_a[0], rho_b[0]
            ex_a, vx_a, fx_a, _ = eval_x_acpbe1('GGA_X_acPBE', 2*rho_a, spin=0, relativity=relativity, 
                                                    deriv=deriv, omega=omega, verbose=verbose)
            ex_b, vx_b, fx_b, _ = eval_x_acpbe1('GGA_X_acPBE', 2*rho_b, spin=0, relativity=relativity, 
                                                    deriv=deriv, omega=omega, verbose=verbose)
            n_total = n_a + n_b
            ex = np.zeros_like(n_total)
            for i in range(len(n_total)):
                ex[i] = n_a[i] / n_total[i] * ex_a[i] + n_b[i] / n_total[i] * ex_b[i]
            if deriv > 0:
                vrho_a, vsigma_a = vx_a
                vrho_b, vsigma_b = vx_b
                # vrho: (N, 2) vsigma: (N, 3)            
                vrho = np.array([
                    vrho_a, 
                    vrho_b
                    ]).T
                vsigma = np.array([
                    2 * vsigma_a, 
                    np.zeros_like(n_total), 
                    2 * vsigma_b
                    ]).T
                vx = (vrho, vsigma)
            if deriv > 1:
                v2rho2_a, v2rhosigma_a, v2sigma2_a = fx_a
                v2rho2_b, v2rhosigma_b, v2sigma2_b = fx_b
                # v2rho2: (N, 3) v2rhosigma: (N, 6) v2sigma2: (N, 6)
                v2rho2 = np.array([
                    2 * v2rho2_a,           # rho_a rho_a
                    np.zeros_like(n_total), # rho_a rho_b
                    2 * v2rho2_b            # rho_b rho_b
                    ]).T                
                v2rhosigma = np.array([
                    4 * v2rhosigma_a,       # rho_a sigma_aa
                    np.zeros_like(n_total), # rho_a sigma_ab
                    np.zeros_like(n_total), # rho_a sigma_bb
                    np.zeros_like(n_total), # rho_b sigma_aa
                    np.zeros_like(n_total), # rho_b sigma_ab 
                    4 * v2rhosigma_b
                    ]).T    # rho_b sigma_bb
                v2sigma2 = np.array([
                    8 * v2sigma2_a,         # sigma_aa sigma_aa
                    np.zeros_like(n_total),        # sigma_aa sigma_ab
                    np.zeros_like(n_total),        # sigma_aa sigma_bb
                    np.zeros_like(n_total),        # sigma_ab sigma_aa
                    np.zeros_like(n_total),        # sigma_ab sigma_ab
                    8 * v2sigma2_b
                    ]).T      # sigma_bb sigma_bb
                fx = (v2rho2, v2rhosigma, v2sigma2)
        return ex, vx, fx, kx


    def eval_c_acpbe(xc_code, rho, spin, relativity=0, deriv=1, omega=None, verbose=None):
        ec, vc, fc, kc = dft.libxc.eval_xc('GGA_C_PBE', rho, spin, relativity, deriv, omega, verbose)
        return ec, vc, fc, kc


    def eval_xc_acpbe1(xc_code, rho, spin, relativity=0, deriv=1, omega=None, verbose=None):
        ex, vx, fx, kx = eval_x_acpbe1('GGA_X_acPBE', rho, spin, relativity, deriv, omega, verbose)
        ec, vc, fc, kc = eval_c_acpbe('GGA_C_PBE', rho, spin, relativity, deriv, omega, verbose)

        exc = ex + ec
        vxc, fxc, kxc = None, None, None
        if deriv > 0:
            vrho_x, vsigma_x = vx # type: ignore
            vrho_c, vsigma_c = vc
            vrho = vrho_x + vrho_c
            vsigma = vsigma_x + vsigma_c
            vxc = (vrho, vsigma)
        if deriv > 1:
            v2rho2_x, v2rhosigma_x, v2sigma2_x = fx # type: ignore
            v2rho2_c, v2rhosigma_c, v2sigma2_c = fc
            v2rho2 = v2rho2_x + v2rho2_c
            v2rhosigma = v2rhosigma_x + v2rhosigma_c
            v2sigma2 = v2sigma2_x + v2sigma2_c
            fxc = (v2rho2, v2rhosigma, v2sigma2)

        return exc, vxc, fxc, kxc

    return eval_xc_acpbe1


def acpbe2(alpha, Cx):
    def eval_x_acpbe2(xc_code, rho, spin, relativity=0, deriv=1, omega=None, verbose=None):
        ex, vx, fx, kx = None, None, None, None
        if spin == 0:  # ===== 非自旋极化情况 =====
            n = rho[0]
            n += 1e-14
            rho_aux = np.copy(rho)
            rho_aux[0] = n ** (alpha * 3.0 / 4.0)
            ex_aux, vx_aux, fx_aux, _ = dft.libxc.eval_xc('GGA_X_PBE', rho_aux, spin, relativity, deriv, omega, verbose)
            factor = n ** (alpha * 3.0 / 4.0 - 1)
            ex = Cx * factor * ex_aux
            
            if deriv > 0:
                vrho_aux, vsigma_aux = vx_aux
                vrho = Cx * alpha * 3.0 / 4.0 * factor * vrho_aux
                vsigma = Cx * vsigma_aux
                vx = (vrho, vsigma)
            if deriv > 1:
                v2rho2_aux, v2rhosigma_aux, v2sigma2_aux = fx_aux
                v2rho2 = Cx * (alpha * 3.0 / 4.0 * factor) ** 2 * v2rho2_aux + \
                         Cx * (alpha * 3.0 / 4.0) * (alpha * 3.0 / 4.0 - 1) * factor / n * vrho_aux
                v2rhosigma = Cx * alpha * 3.0 / 4.0 * factor * v2rhosigma_aux
                v2sigma2 = Cx * v2sigma2_aux
                fx = (v2rho2, v2rhosigma, v2sigma2)

        else:  # ===== 自旋极化情况 =====
            rho_a, rho_b = rho[0], rho[1]
            n_a, n_b = rho_a[0], rho_b[0]
            ex_a, vx_a, fx_a, _ = eval_x_acpbe2('GGA_X_acPBE', 2*rho_a, spin=0, relativity=relativity, 
                                                    deriv=deriv, omega=omega, verbose=verbose)
            ex_b, vx_b, fx_b, _ = eval_x_acpbe2('GGA_X_acPBE', 2*rho_b, spin=0, relativity=relativity, 
                                                    deriv=deriv, omega=omega, verbose=verbose)
            n_total = n_a + n_b
            ex = np.zeros_like(n_total)
            for i in range(len(n_total)):
                ex[i] = n_a[i] / n_total[i] * ex_a[i] + n_b[i] / n_total[i] * ex_b[i]
            if deriv > 0:
                vrho_a, vsigma_a = vx_a
                vrho_b, vsigma_b = vx_b
                # vrho: (N, 2) vsigma: (N, 3)            
                vrho = np.array([
                    vrho_a, 
                    vrho_b
                    ]).T
                vsigma = np.array([
                    2 * vsigma_a,
                    np.zeros_like(n_total),
                    2 * vsigma_b
                    ]).T
                vx = (vrho, vsigma)
            if deriv > 1:
                v2rho2_a, v2rhosigma_a, v2sigma2_a = fx_a
                v2rho2_b, v2rhosigma_b, v2sigma2_b = fx_b
                # v2rho2: (N, 3) v2rhosigma: (N, 6) v2sigma2: (N, 6)
                v2rho2 = np.array([
                    2 * v2rho2_a,           # rho_a rho_a
                    np.zeros_like(n_total), # rho_a rho_b
                    2 * v2rho2_b            # rho_b rho_b
                    ]).T                
                v2rhosigma = np.array([
                    4 * v2rhosigma_a,       # rho_a sigma_aa
                    np.zeros_like(n_total), # rho_a sigma_ab
                    np.zeros_like(n_total), # rho_a sigma_bb
                    np.zeros_like(n_total), # rho_b sigma_aa
                    np.zeros_like(n_total), # rho_b sigma_ab 
                    4 * v2rhosigma_b
                    ]).T    # rho_b sigma_bb
                v2sigma2 = np.array([
                    8 * v2sigma2_a,         # sigma_aa sigma_aa
                    np.zeros_like(n_total),        # sigma_aa sigma_ab
                    np.zeros_like(n_total),        # sigma_aa sigma_bb
                    np.zeros_like(n_total),        # sigma_ab sigma_aa
                    np.zeros_like(n_total),        # sigma_ab sigma_ab
                    8 * v2sigma2_b
                    ]).T      # sigma_bb sigma_bb
                fx = (v2rho2, v2rhosigma, v2sigma2)

        return ex, vx, fx, kx


    def eval_c_acpbe2(xc_code, rho, spin, relativity=0, deriv=1, omega=None, verbose=None):
        ec, vc, fc, kc = dft.libxc.eval_xc('GGA_C_PBE', rho, spin, relativity, deriv, omega, verbose)
        return ec, vc, fc, kc


    def eval_xc_acpbe2(xc_code, rho, spin, relativity=0, deriv=1, omega=None, verbose=None):
        ex, vx, fx, kx = eval_x_acpbe2('GGA_X_acPBE', rho, spin, relativity, deriv, omega, verbose)
        ec, vc, fc, kc = eval_c_acpbe2('GGA_C_PBE', rho, spin, relativity, deriv, omega, verbose)

        exc = ex + ec
        vxc, fxc, kxc = None, None, None
        if deriv > 0:
            vrho_x, vsigma_x = vx # type: ignore
            vrho_c, vsigma_c = vc
            vrho = vrho_x + vrho_c
            vsigma = vsigma_x + vsigma_c
            vxc = (vrho, vsigma)
        if deriv > 1:
            v2rho2_x, v2rhosigma_x, v2sigma2_x = fx # type: ignore
            v2rho2_c, v2rhosigma_c, v2sigma2_c = fc
            v2rho2 = v2rho2_x + v2rho2_c
            v2rhosigma = v2rhosigma_x + v2rhosigma_c
            v2sigma2 = v2sigma2_x + v2sigma2_c
            fxc = (v2rho2, v2rhosigma, v2sigma2)

        return exc, vxc, fxc, kxc

    return eval_xc_acpbe2


def acscan(alpha, Cx):
    def eval_x_acscan(xc_code, rho, spin, relativity=0, deriv=1, omega=None, verbose=None):
        ex_ac, vx_ac, fx_ac, kx_ac = None, None, None, None
        if spin == 0:  # ===== 非自旋极化情况 =====
            n = rho[0]
            n += 1e-14
            rho_aux = np.copy(rho)
            rho_aux[0] = n ** (alpha * 3.0 / 4.0)
            ex_aux, vx_aux, fx_aux, _ = dft.libxc.eval_xc('MGGA_X_SCAN', rho_aux, spin, relativity, deriv, omega, verbose)
            factor = n ** (alpha * 3.0 / 4.0 - 1)
            ex_ac = Cx * factor * ex_aux
            
            if deriv > 0:
                vrho_aux, vsigma_aux, _, vtau_aux = vx_aux  # vrho, vsigma, vlapl, vtau
                vrho = Cx * alpha * 3.0 / 4.0 * factor * vrho_aux
                vsigma = Cx * vsigma_aux
                vtau = Cx * vtau_aux
                vx_ac = (vrho, vsigma, None, vtau)
            if deriv > 1:
                # (v2rho2, v2rhosigma, v2sigma2, v2lapl2, v2tau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)
                v2rho2_aux, v2rhosigma_aux, v2sigma2_aux, _, v2tau2_aux, _, v2rhotau_aux, _, _, v2sigmatau_aux = fx_aux 
                v2rho2 = Cx * (alpha * 3.0 / 4.0 * factor) ** 2 * v2rho2_aux + \
                         Cx * (3.0 * alpha / 4.0) * (3.0 * alpha / 4.0 - 1) * n ** (alpha * 3.0 / 4.0 - 2) * vrho_aux
                v2rhosigma = Cx * alpha * 3.0 / 4.0 * factor * v2rhosigma_aux
                v2rhotau = Cx * alpha * 3.0 / 4.0 * factor * v2rhotau_aux
                v2sigma2 = Cx * v2sigma2_aux
                v2sigmatau = Cx * v2sigmatau_aux
                v2tau2 = Cx * v2tau2_aux
                fx_ac = [v2rho2, v2rhosigma, v2sigma2, None, v2tau2, None, v2rhotau, None, None, v2sigmatau]

        else:  # ===== 自旋极化情况 =====
            rho_a, rho_b = rho[0], rho[1]
            n_a, n_b = rho_a[0], rho_b[0]
            ex_a, vx_a, fx_a, _ = eval_x_acscan('MGGA_X_acSCAN', 2*rho_a, spin=0, relativity=relativity, 
                                                    deriv=deriv, omega=omega, verbose=verbose)
            ex_b, vx_b, fx_b, _ = eval_x_acscan('MGGA_X_acSCAN', 2*rho_b, spin=0, relativity=relativity, 
                                                    deriv=deriv, omega=omega, verbose=verbose)
            n_total = n_a + n_b
            ex_ac = np.zeros_like(n_total)
            for i in range(len(n_total)):
                ex_ac[i] = n_a[i] / n_total[i] * ex_a[i] + n_b[i] / n_total[i] * ex_b[i]
            if deriv > 0:
                vrho_a, vsigma_a, _, vtau_a = vx_a
                vrho_b, vsigma_b, _, vtau_b = vx_b
                # vrho: (N, 2) vsigma: (N, 3)            
                vrho = np.array([vrho_a, vrho_b]).T
                vsigma = np.array(
                    [2 * vsigma_a,       # sigma_aa
                     np.zeros_like(n_total),    # sigma_ab
                     2 * vsigma_b]).T    # sigma_bb
                vtau = np.array([vtau_a, vtau_b]).T
                vx_ac = (vrho, vsigma, None, vtau)
            if deriv > 1:
                # (v2rho2, v2rhosigma, v2sigma2, v2lapl2, v2tau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)
                v2rho2_a, v2rhosigma_a, v2sigma2_a, _, v2tau2_a, _, v2rhotau_a, _, _, v2sigmatau_a = fx_a
                v2rho2_b, v2rhosigma_b, v2sigma2_b, _, v2tau2_b, _, v2rhotau_b, _, _, v2sigmatau_b = fx_b
                v2rho2 = np.array([
                    2 * v2rho2_a,                 # rho_a rho_a
                    np.zeros_like(n_total),        # rho_a rho_b
                    2 * v2rho2_b
                    ]).T              # rho_b rho_b
                v2rhosigma = np.array([
                    4 * v2rhosigma_a,             # rho_a sigma_aa
                    np.zeros_like(n_total),        # rho_a sigma_ab
                    np.zeros_like(n_total),        # rho_a sigma_bb
                    np.zeros_like(n_total),        # rho_b sigma_aa
                    np.zeros_like(n_total),        # rho_b sigma_ab 
                    4 * v2rhosigma_b
                    ]).T          # rho_b sigma_bb
                v2rhotau = np.array([
                    2 * v2rhotau_a,               # rho_a, tau_a
                    np.zeros_like(n_a),            # rho_a, tau_b
                    np.zeros_like(n_a),            # rho_b, tau_a
                    2 * v2rhotau_b
                    ]).T            # rho_b, tau_b
                v2sigma2 = np.array([
                    8 * v2sigma2_a,               
                    np.zeros_like(n_total),        
                    np.zeros_like(n_total),       
                    np.zeros_like(n_total),       
                    np.zeros_like(n_total),
                    8 * v2sigma2_b   
                    ]).T            
                v2sigmatau = np.array([
                    4 * v2sigmatau_a,             
                    np.zeros_like(n_a),            
                    np.zeros_like(n_a),            
                    np.zeros_like(n_a),            
                    np.zeros_like(n_a),            
                    4 * v2sigmatau_b
                    ]).T         
                v2tau2 = np.array([
                    2 * v2tau2_a,                 # tau_a, tau_a
                    np.zeros_like(n_a),            # tau_a, tau_b
                    2 * v2tau2_b
                    ]).T             

                fx_ac = [v2rho2, v2rhosigma, v2sigma2, None, v2tau2, None, v2rhotau, None, None, v2sigmatau]

        return ex_ac, vx_ac, fx_ac, kx_ac


    def eval_c_acscan(xc_code, rho, spin, relativity=0, deriv=1, omega=None, verbose=None):
        ec, vc, fc, kc = dft.libxc.eval_xc('MGGA_C_SCAN', rho, spin, relativity, deriv, omega, verbose)
        return ec, vc, fc, kc


    def eval_xc_acscan(xc_code, rho, spin, relativity=0, deriv=1, omega=None, verbose=None):
        ex, vx, fx, kx = eval_x_acscan('MGGA_X_SCAN', rho, spin, relativity, deriv, omega, verbose)
        ec, vc, fc, kc = eval_c_acscan('MGGA_C_SCAN', rho, spin, relativity, deriv, omega, verbose)
        # ex, vx, fx, kx = dft.libxc.eval_xc('MGGA_X_SCAN', rho, spin, relativity, deriv, omega, verbose)
        # ec, vc, fc, kc = dft.libxc.eval_xc('MGGA_C_SCAN', rho, spin, relativity, deriv, omega, verbose)
        
        exc = ex + ec
        vxc, fxc, kxc = None, None, None
        if deriv > 0:
            vrho_x, vsigma_x, _, vtau_x = vx # type: ignore
            vrho_c, vsigma_c, _, vtau_c = vc
            vrho = vrho_x + vrho_c
            vsigma = vsigma_x + vsigma_c
            vtau = vtau_x + vtau_c
            vxc = (vrho, vsigma, None, vtau)
        if deriv > 1:
            # (v2rho2, v2rhosigma, v2sigma2, v2lapl2, v2tau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)
            v2rho2_x, v2rhosigma_x, v2sigma2_x, _, v2tau2_x, _, v2rhotau_x, _, _, v2sigmatau_x = fx # type: ignore
            v2rho2_c, v2rhosigma_c, v2sigma2_c, _, v2tau2_c, _, v2rhotau_c, _, _, v2sigmatau_c = fc
            v2rho2 = v2rho2_x + v2rho2_c
            v2rhosigma = v2rhosigma_x + v2rhosigma_c
            v2rhotau = v2rhotau_x + v2rhotau_c
            v2sigma2 = v2sigma2_x + v2sigma2_c
            v2sigmatau = v2sigmatau_x + v2sigmatau_c
            v2tau2 = v2tau2_x + v2tau2_c
            fxc = (v2rho2, v2rhosigma, v2sigma2, None, v2tau2, None, v2rhotau, None, None, v2sigmatau)

        return exc, vxc, fxc, kxc

    return eval_xc_acscan


def test1_aclda():
    r_grid = np.linspace(1e-6, 10, 10)
    n = np.exp(-r_grid / 2)

    rho_pol = np.array([n, n])
    rho_unpol = n
    spin = 0
    if spin == 0:
        rho = rho_unpol
    else:
        rho = rho_pol

    ex_aux, vx_aux, fx_aux, _ = dft.libxc.eval_xc('LDA_X', rho, spin=spin, deriv=2)
    ec_aux, vc_aux, fc_aux, _ = dft.libxc.eval_xc('LDA_C_VWN', rho, spin=spin, deriv=2)
    exc_aux, vxc_aux, fxc_aux, _ = dft.libxc.eval_xc('LDA, VWN', rho, spin=spin, deriv=2)
    
    eval_xc_aclda = aclda1(alpha=4/3, Cx=1.0)
    exc, vxc, fxc, _ = eval_xc_aclda('LDA', rho, spin=spin, deriv=2)
    
    print("fxc_aux:\n", fxc_aux)
    print("fxc:\n", fxc)
    # print("ec_aux:\n", ec_aux)
    # print("vx_aux:\n", vx_aux)
    # print("vc_aux:\n", vc_aux)
    # print("fx_aux:\n", fx_aux)
    # print("fc_aux:\n", fc_aux)

def test2_aclda():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'
    mol.basis = 'def2-SVP'
    mol.spin = 0
    mol.charge = 0
    mol.build()

    mf1 = dft.KS(mol)
    mf1.xc = "LDA, VWN"
    mf1 = mf1.newton()
    mf1.kernel()

    mf2 = dft.KS(mol)
    eval_xc_aclda = aclda2(alpha=4/3, Cx=1.0)
    mf2 = mf2.define_xc_(eval_xc_aclda, 'LDA', hyb=0)
    mf2 = mf2.newton()
    # mf2.max_cycle = 300
    mf2.kernel()

def test_acpbe1():
    r_grid = np.linspace(1, 10, 5)
    n = np.exp(-r_grid / 2)

    rho_unpol = np.array([n, n/np.sqrt(3), n/np.sqrt(3), n/np.sqrt(3)])
    rho_pol = np.array([ 2 * rho_unpol, 2 * rho_unpol])
    spin = 1
    if spin == 0:
        rho = rho_unpol
    else:
        rho = rho_pol

    exc_aux, vxc_aux, fxc_aux, _ = dft.libxc.eval_xc('PBE', rho, spin=spin, deriv=2)
    eval_xc_acpbe1 = acpbe1(alpha=4/3, Cx=1.0)
    exc, vxc, fxc, _ = eval_xc_acpbe1('GGA', rho, spin=spin, deriv=2)
    
    for i in range(3):
        print(f"fxc{i}:\n", fxc[i]-fxc_aux[i])
    pass

def test1_acscan():
    r_grid = np.linspace(1, 10, 5)
    n = np.exp(-r_grid / 2)

    rho_unpol = np.array([n, n/np.sqrt(3), n/np.sqrt(3), n/np.sqrt(3), n/2])
    rho_pol = np.array([ 2 * rho_unpol, 2 * rho_unpol])
    spin = 1
    if spin == 0:
        rho = rho_unpol
    else:
        rho = rho_pol

    ex_aux, vx_aux, fx_aux, _ = dft.libxc.eval_xc('MGGA_X_SCAN', rho, spin=spin, deriv=2)
    ec_aux, vc_aux, fc_aux, _ = dft.libxc.eval_xc('MGGA_C_SCAN', rho, spin=spin, deriv=2)
    exc_aux, vxc_aux, fxc_aux, _ = dft.libxc.eval_xc('SCAN', rho, spin=spin, deriv=2)
    
    eval_xc_acscan = acscan(alpha=4/3, Cx=1.0)
    exc, vxc, fxc, _ = eval_xc_acscan('MGGA', rho, spin=spin, deriv=2)
    
    for i in range(10):
        if fxc[i] is not None:
            print(f"fxc{i}:\n", np.max(abs(fxc[i]-fxc_aux[i])))
        else:
            print(f"fxc{i}:\n", fxc[i]==fxc_aux[i])
        # print(f"fxc_aux{i}:\n", fxc_aux[i])
    # print("ec_aux:\n", ec_aux)
    # print("vx_aux:\n", vx_aux)
    # print("vc_aux:\n", vc_aux)
    # print("fx_aux:\n", fx_aux)
    # print("fc_aux:\n", fc_aux)


def test2_acscan():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'
    mol.basis = 'def2-qzvppd'
    mol.spin = 0
    mol.charge = 0
    mol.build()

    mf1 = dft.KS(mol)
    mf1.xc = "SCAN"
    mf1.verbose = 0
    mf1 = mf1.newton()
    mf1.max_cycle = 300
    mf1.level_shift = 0.2
    mf1.kernel()
    dm = mf1.make_rdm1()
    print(mf1.e_tot)

    mf2 = dft.KS(mol)
    eval_xc_acscan = acscan(alpha=1.5, Cx=1.0)
    mf2 = mf2.define_xc_(eval_xc_acscan, 'MGGA', hyb=0)
    mf2.verbose = 4
    # mf2 = mf2.newton()
    mf2.max_cycle = 300
    mf2.level_shift = 0.2
    mf2.kernel()
    print(mf2.e_tot)


def main():
    # test()
    # test_acpbe1()
    # test2_aclda()
    test1_acscan()


if __name__ == '__main__':
    main()

