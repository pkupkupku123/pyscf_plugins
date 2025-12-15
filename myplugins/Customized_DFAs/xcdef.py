from collections import namedtuple

import numpy as np
from pyscf import dft  # type: ignore

grids = np.ndarray | None


class ModifiedXC:
    XC = namedtuple("XC", ["x", "c", "level"])
    BUILTIN_XC = namedtuple("BUILTIN_XC", ["xc_code", "x_mode", "c_mode"])

    FLAG_ORIGIN = 0
    FLAG_MODIFY_RHO = 1
    FLAG_MODIFY_RHO2 = 2

    XC_MODE_DICT = {
        "ORIGIN": FLAG_ORIGIN,
        "MODIFY-RHO": FLAG_MODIFY_RHO,
        "MODIFY-RHO2": FLAG_MODIFY_RHO2,
    }
    SUPPORTED_X_MODE = XC_MODE_DICT.keys()
    SUPPORTED_C_MODE = ["ORIGIN"]

    XC_DICT = {
        # LDA
        "LDA": XC("LDA_X", "LDA_C_VWN", "LDA"),
        # GGA
        "PBE": XC("GGA_X_PBE", "GGA_C_PBE", "GGA"),
        "BPBE": XC("GGA_X_B88", "GGA_C_PBE", "GGA"),
        "BLYP": XC("GGA_X_B88", "GGA_C_LYP", "GGA"),
        "BP86": XC("GGA_X_B88", "GGA_C_P86", "GGA"),
        "BPW91": XC("GGA_X_B88", "GGA_C_PW91", "GGA"),
        # MGGA
        "SCAN": XC("MGGA_X_SCAN", "MGGA_C_SCAN", "MGGA"),
        "R2SCAN": XC("MGGA_X_R2SCAN", "MGGA_C_R2SCAN", "MGGA"),
        "PKZB": XC("MGGA_X_PKZB", "MGGA_C_PKZB", "MGGA"),
        "TPSS": XC("MGGA_X_TPSS", "MGGA_C_TPSS", "MGGA"),
    }
    SUPPORTED_XC_CODE = XC_DICT.keys()

    BUILTIN_XC_DICT = {
        # acLDA
        "ACLDA": BUILTIN_XC("LDA", "modify-rho", "origin"),
        "ACLDA1": BUILTIN_XC("LDA", "modify-rho", "origin"),
        "ACLDA2": BUILTIN_XC("LDA", "modify-rho2", "origin"),
        # acGGA1
        "ACPBE1": BUILTIN_XC("PBE", "modify-rho", "origin"),
        "ACBPBE1": BUILTIN_XC("BPBE", "modify-rho", "origin"),
        "ACBLYP1": BUILTIN_XC("BLYP", "modify-rho", "origin"),
        "ACBP861": BUILTIN_XC("BP86", "modify-rho", "origin"),
        "ACBPW911": BUILTIN_XC("BPW91", "modify-rho", "origin"),
        # acMGGA1
        "ACSCAN1": BUILTIN_XC("SCAN", "modify-rho", "origin"),
        "ACR2SCAN1": BUILTIN_XC("R2SCAN", "modify-rho", "origin"),
        "ACPKZB1": BUILTIN_XC("PKZB", "modify-rho", "origin"),
        "ACTPSS1": BUILTIN_XC("TPSS", "modify-rho", "origin"),
        # acGGA2
        "ACPBE2": BUILTIN_XC("PBE", "modify-rho2", "origin"),
        "ACBPBE2": BUILTIN_XC("BPBE", "modify-rho2", "origin"),
        "ACBLYP2": BUILTIN_XC("BLYP", "modify-rho2", "origin"),
        "ACBP862": BUILTIN_XC("BP86", "modify-rho2", "origin"),
        "ACBPW912": BUILTIN_XC("BPW91", "modify-rho2", "origin"),
        # acMGGA2
        "ACSCAN2": BUILTIN_XC("SCAN", "modify-rho2", "origin"),
        "ACR2SCAN2": BUILTIN_XC("R2SCAN", "modify-rho2", "origin"),
        "ACPKZB2": BUILTIN_XC("PKZB", "modify-rho2", "origin"),
        "ACTPSS2": BUILTIN_XC("TPSS", "modify-rho2", "origin"),
    }
    SUPPORTED_BUILTIN_XC = BUILTIN_XC_DICT.keys()

    MAX_TUPLE_DEPTH = 3

    SINGLET_SPIN = 0

    # TODO: check all the u_factors

    U_FACTOR_RHO = 2

    U_FACTOR_VRHO = 1
    U_FACTOR_VSIGMA = 2
    U_FACTOR_VLAPL = 1
    U_FACTOR_VTAU = 1

    U_FACTOR_V2RHO2 = 2
    U_FACTOR_V2RHOSIGMA = 4
    U_FACTOR_V2SIGMA2 = 8
    U_FACTOR_V2LAPL2 = 2
    U_FACTOR_V2TAU2 = 2
    U_FACTOR_V2RHOLAPL = 2
    U_FACTOR_V2RHOTAU = 2
    U_FACTOR_V2LAPPTAU = 2
    U_FACTOR_V2SIGMALAPL = 4
    U_FACTOR_V2SIGMATAU = 4

    U_FACTOR_V3RHO3 = 4
    U_FACTOR_V3RHO2SIGMA = 8
    U_FACTOR_V3RHOSIGMA2 = 16
    U_FACTOR_V3SIGMA3 = 32
    U_FACTOR_V3RHO2LAPL = 4
    U_FACTOR_V3RHO2TAU = 4
    U_FACTOR_V3RHOSIGMALAPL = 8
    U_FACTOR_V3RHOSIGMATAU = 8
    U_FACTOR_V3RHOLAPL2 = 4
    U_FACTOR_V3RHOLAPLTAU = 4
    U_FACTOR_V3RHOTAU2 = 4
    U_FACTOR_V3SIGMA2LAPL = 16
    U_FACTOR_V3SIGMA2TAU = 16
    U_FACTOR_V3SIGMALAPL2 = 8
    U_FACTOR_V3SIGMALAPLTAU = 8
    U_FACTOR_V3SIGMATAU2 = 8
    U_FACTOR_V3LAPL3 = 4
    U_FACTOR_V3LAPL2TAU = 4
    U_FACTOR_V3LAPLTAU2 = 4
    U_FACTOR_V3TAU3 = 4

    U_FACTORS_VXC = (U_FACTOR_VRHO, U_FACTOR_VSIGMA, U_FACTOR_VLAPL, U_FACTOR_VTAU)
    U_FACTORS_FXC = (
        U_FACTOR_V2RHO2,
        U_FACTOR_V2RHOSIGMA,
        U_FACTOR_V2SIGMA2,
        U_FACTOR_V2LAPL2,
        U_FACTOR_V2TAU2,
        U_FACTOR_V2RHOLAPL,
        U_FACTOR_V2RHOTAU,
        U_FACTOR_V2LAPPTAU,
        U_FACTOR_V2SIGMALAPL,
        U_FACTOR_V2SIGMATAU,
    )
    U_FACTORS_KXC = (
        U_FACTOR_V3RHO3,
        U_FACTOR_V3RHO2SIGMA,
        U_FACTOR_V3RHOSIGMA2,
        U_FACTOR_V3SIGMA3,
        U_FACTOR_V3RHO2LAPL,
        U_FACTOR_V3RHO2TAU,
        U_FACTOR_V3RHOSIGMALAPL,
        U_FACTOR_V3RHOSIGMATAU,
        U_FACTOR_V3RHOLAPL2,
        U_FACTOR_V3RHOLAPLTAU,
        U_FACTOR_V3RHOTAU2,
        U_FACTOR_V3SIGMA2LAPL,
        U_FACTOR_V3SIGMA2TAU,
        U_FACTOR_V3SIGMALAPL2,
        U_FACTOR_V3SIGMALAPLTAU,
        U_FACTOR_V3SIGMATAU2,
        U_FACTOR_V3LAPL3,
        U_FACTOR_V3LAPL2TAU,
        U_FACTOR_V3LAPLTAU2,
        U_FACTOR_V3TAU3,
    )

    # TODO: check all the lens

    LEN_VXC = 4
    LEN_VRHO_U = 2
    LEN_VSIGMA_U = 3
    LEN_VLAPL_U = 2
    LEN_VTAU_U = 2
    LENS_VXC_U = (LEN_VRHO_U, LEN_VSIGMA_U, LEN_VLAPL_U, LEN_VTAU_U)

    LEN_FXC = 10
    LEN_V2RHO2_U = 3
    LEN_V2RHOSIGMA_U = 6
    LEN_V2SIGMA2_U = 6
    LEN_V2LAPL2_U = 3
    LEN_V2TAU2_U = 3
    LEN_V2RHOLAPL_U = 4
    LEN_V2RHOTAU_U = 4
    LEN_V2LAPPTAU_U = 4
    LEN_V2SIGMALAPL_U = 6
    LEN_V2SIGMATAU_U = 6
    LENS_FXC_U = (
        LEN_V2RHO2_U,
        LEN_V2RHOSIGMA_U,
        LEN_V2SIGMA2_U,
        LEN_V2LAPL2_U,
        LEN_V2TAU2_U,
        LEN_V2RHOLAPL_U,
        LEN_V2RHOTAU_U,
        LEN_V2LAPPTAU_U,
        LEN_V2SIGMALAPL_U,
        LEN_V2SIGMATAU_U,
    )

    LEN_KXC = 20
    LEN_V3RHO3_U = 4
    LEN_V3RHO2SIGMA_U = 9
    LEN_V3RHOSIGMA2_U = 12
    LEN_V3SIGMA3_U = 10
    LEN_V3RHO2LAPL_U = 6
    LEN_V3RHO2TAU_U = 6
    LEN_V3RHOSIGMALAPL_U = 12
    LEN_V3RHOSIGMATAU_U = 12
    LEN_V3RHOLAPL2_U = 6
    LEN_V3RHOLAPLTAU_U = 8
    LEN_V3RHOTAU2_U = 6
    LEN_V3SIGMA2LAPL_U = 12
    LEN_V3SIGMA2TAU_U = 12
    LEN_V3SIGMALAPL2_U = 9
    LEN_V3SIGMALAPLTAU_U = 12
    LEN_V3SIGMATAU2_U = 9
    LEN_V3LAPL3_U = 4
    LEN_V3LAPL2TAU_U = 6
    LEN_V3LAPLTAU2_U = 6
    LEN_V3TAU3_U = 4
    LENS_KXC_U = (
        LEN_V3RHO3_U,
        LEN_V3RHO2SIGMA_U,
        LEN_V3RHOSIGMA2_U,
        LEN_V3SIGMA3_U,
        LEN_V3RHO2LAPL_U,
        LEN_V3RHO2TAU_U,
        LEN_V3RHOSIGMALAPL_U,
        LEN_V3RHOSIGMATAU_U,
        LEN_V3RHOLAPL2_U,
        LEN_V3RHOLAPLTAU_U,
        LEN_V3RHOTAU2_U,
        LEN_V3SIGMA2LAPL_U,
        LEN_V3SIGMA2TAU_U,
        LEN_V3SIGMALAPL2_U,
        LEN_V3SIGMALAPLTAU_U,
        LEN_V3SIGMATAU2_U,
        LEN_V3LAPL3_U,
        LEN_V3LAPL2TAU_U,
        LEN_V3LAPLTAU2_U,
        LEN_V3TAU3_U,
    )

    def __init__(
        self,
        xc_code: str,
        x_mode: str,
        c_mode: str,
        alpha: float | None = None,
        Cx: float | None = None,
    ):
        self.xc_code = xc_code
        self.x_mode = x_mode
        self.c_mode = c_mode
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = 4/3
        if Cx is not None:
            self.Cx = Cx
        else:
            self.Cx = 1.0

    @property
    def xc_code(self):
        if not hasattr(self, "_xc_code"):
            raise AttributeError("xc_code is not set.")
        return self._xc_code

    @xc_code.setter
    def xc_code(self, value: str):
        value = self.check_and_standarize(value, self.SUPPORTED_XC_CODE)
        self._xc_code = self.XC_DICT[value]

    @property
    def x_mode(self):
        return self._x_mode

    @x_mode.setter
    def x_mode(self, value: str):
        value = self.check_and_standarize(value, self.SUPPORTED_X_MODE)
        self._x_mode = self.XC_MODE_DICT[value]

    @property
    def c_mode(self):
        return self._c_mode

    @c_mode.setter
    def c_mode(self, value: str):
        value = self.check_and_standarize(value, self.SUPPORTED_C_MODE)
        self._c_mode = self.XC_MODE_DICT[value]

    @staticmethod
    def check_and_standarize(value, supported_list):
        value = value.upper().strip().replace(" ", "").replace("_", "-")
        if value not in supported_list:
            raise ValueError(
                f"Invalid value: {value}. Supported values are: {supported_list}"
            )
        return value

    @property
    def alpha(self):
        if not hasattr(self, "_alpha"):
            raise AttributeError("alpha is not set.")
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        if not isinstance(value, (float, int)):
            raise ValueError("alpha must be a float or int.")
        self._alpha = float(value)

    @property
    def Cx(self):
        if not hasattr(self, "_Cx"):
            raise AttributeError("Cx is not set.")
        return self._Cx

    @Cx.setter
    def Cx(self, value: float):
        if not isinstance(value, (float, int)):
            raise ValueError("Cx must be a float or int.")
        self._Cx = float(value)

    @staticmethod
    def sanitize_value(
        x: grids | tuple[grids], copy: bool = False, tuple_depth: int = 0
    ) -> grids | tuple[grids]:
        """Sanitize the input value by replacing nans and infs.

        Args:
            x (grids | tuple[grids]): _description_
            copy (bool, optional): _description_. Defaults to False.
            tuple_depth (int, optional): _description_. Defaults to 0.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            grids | tuple[grids]: _description_
        """
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, copy=copy)
        if isinstance(x, tuple):
            if tuple_depth >= ModifiedXC.MAX_TUPLE_DEPTH:
                raise ValueError("Too deep tuple nesting to handle.")
            return tuple(
                ModifiedXC.sanitize_value(xi, copy=copy, tuple_depth=tuple_depth + 1)
                for xi in x
            )  # type: ignore
        raise ValueError("Input must be a np.ndarray or a tuple of np.ndarray.")

    @staticmethod
    def safe_sum(
        a: grids | tuple[grids] | list[grids],
        b: grids | tuple[grids] | list[grids],
        tuple_depth: int = 0,
    ) -> grids | tuple[grids] | list[grids]:
        """Safely sum two values, handling None types and automatically unpacking tuples.

        Args:
            a (_type_): _description_
            b (_type_): _description_
        """

        if (a is None) and (b is None):
            return None
        if a is None:
            return b
        if b is None:
            return a
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            if tuple_depth >= ModifiedXC.MAX_TUPLE_DEPTH:
                raise ValueError("Too deep tuple nesting to handle.")
            # if len(a) != len(b):
            #     if
            return tuple(
                ModifiedXC.safe_sum(x, y, tuple_depth=tuple_depth + 1)
                for x, y in zip(a, b)
            )  # type: ignore
        return a + b  # type: ignore

    @staticmethod
    def normalize_tuple_size(x: tuple, length: int) -> tuple:
        """Add None to the end of a tuple until it reaches the specified length.

        Args:
            x (tuple): The original tuple.
            length (int): The desired length of the tuple.

        Returns:
            tuple: The modified tuple with None added to the end if necessary.
        """
        if not isinstance(x, tuple):
            raise ValueError("Input must be a tuple.")
        if len(x) >= length:
            return x[:length]
        return x + (None,) * (length - len(x))

    @staticmethod
    def concatenate_spin(a: np.ndarray, b: np.ndarray, length: int) -> np.ndarray:
        """Concatenate two values along the last dimension with zeros added to inner.

        Args:
            a (np.ndarray): The first grid.
            b (np.ndarray): The second grid.
            length (int): The desired length of the concatenated grid.

        Returns:
            np.ndarray: (*, length) with
                        [:, 0]  -> a
                        [:, 1:length] -> zeros
                        [:, length] -> b
        """
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            raise ValueError("Input must be a np.ndarray.")
        if a.ndim < 1 or b.ndim < 1:
            raise ValueError("Input must have at least one dimension.")
        if a.shape[:-1] != b.shape[:-1]:
            raise ValueError(
                "Input must have the same shape except for the last dimension."
            )
        if length < 2:
            raise ValueError("Length must be at least 2.")

        result = np.zeros((a.size, length), dtype=a.dtype)
        result[..., 0] = a
        result[..., -1] = b
        return result

    @staticmethod
    def all_concatenate(
        a: tuple[grids], b: tuple[grids], length: tuple[int], factors: tuple[int]
    ) -> tuple[grids]:
        """Concatenate two tuples of grids along the last dimension with zeros added to inner.

        Args:
            a (tuple[grids]): The first tuple of grids.
            b (tuple[grids]): The second tuple of grids.
            length (tuple[int]): The desired lengths of the concatenated grids.

        Returns:
            tuple[grids]: The concatenated tuple of grids.
        """
        if not isinstance(a, tuple) or not isinstance(b, tuple):
            raise ValueError("Input must be a tuple.")
        if len(a) != len(b) or len(a) != len(length):
            raise ValueError("Input tuples must have the same length as length tuple.")

        return tuple(
            ModifiedXC.concatenate_spin(x, y, length_val) * factor
            if x is not None and y is not None
            else None
            for x, y, length_val, factor in zip(a, b, length, factors)  # type: ignore
        )

    @classmethod
    def builtin(cls, xc_str: str):
        xc_str = cls.check_and_standarize(xc_str, cls.SUPPORTED_BUILTIN_XC)
        xc_info = cls.BUILTIN_XC_DICT[xc_str]
        return cls(xc_info.xc_code, xc_info.x_mode, xc_info.c_mode)

    def eval_xc(
        self,
        xc_code,
        rho,
        spin,
        relativity=0,
        deriv=1,
        omega=None,
        verbose=None,
    ):
        """Evaluate the exchange-correlation energy and its derivatives.

        Args & Returns: See `pyscf.dft.libxc.eval_xc()` for details.

        [!NOTE]:
        (`a` for alpha spin, `b` for beta spin,
         `R` for spin-restricted, `U` for spin-unrestricted)
        + input shape of rho:
            + `(n_grids,)` for R-LDA
            + `(4, n_grids)` for R-GGA -> [rho, dx, dy, dz]
            + `(5, n_grids)` for R-MGGA -> [rho, dx, dy, dz, tau]
            + `(6, n_grids)` for R-MGGA(with_lapl) -> [rho, dx, dy, dz, tau, lapl]
            + `(2, n_grids)` for U-LDA -> [rho_a, rho_b]
            + `(2, 4, n_grids)` for U-GGA -> [[rho_a, dx_a, dy_a, dz_a],
                                              [rho_b, dx_b, dy_b, dz_b]]
            + `(2, 5, n_grids)` for U-MGGA -> [[rho_a, dx_a, dy_a, dz_a, tau_a],
                                               [rho_b, dx_b, dy_b, dz_b, tau_b]]
            + `(2, 6, n_grids)` for U-MGGA(with_lapl) -> [[rho_a, dx_a, dy_a, dz_a, tau_a, lapl_a],
                                                          [rho_b, dx_b, dy_b, dz_b, tau_b, lapl_b]]
        + return shape of exc, vxc, fxc, kxc:
            + exc
                + `(n_grids,)` for both R- and U- case
            + vxc
                + `vxc = (vrho, vsigma, vlapl, vtau)`
                + `None` for deriv < 1
                + for R- case, `(n_grids,)` for each of the above
                + for U- case:
                    + vrho
                        + `(n_grids, 2)` -> [vrho_a, vrho_b].T
                    + vsigma
                        + `None` for LDA
                        + `(n_grids, 3)` -> [vsigma_aa, vsigma_ab, vsigma_bb].T
                    + vlapl
                        + `None` for LDA and GGA
                        + `(n_grids, 2)` -> [vlapl_a, vlapl_b].T
                    + vtau
                        + `None` for LDA and GGA
                        + `(n_grids, 2)` -> [vtau_a, vtau_b].T
            + fxc
                + `fxc = (v2rho2, v2rhosigma, v2sigma2, v2lapl2, v2tau2, v2rholapl,
                         v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)`
                + `None` for deriv < 2
                + for R- case, `(n_grids,)` for each of the above
                + for U- case:
                    + v2rho2
                        + `(n_grids, 3)` -> [v2rho2_aa, v2rho2_ab, v2rho2_bb].T
                    + v2rhosigma
                        + `None` for LDA
                        + `(n_grids, 6)` -> [v2rhosigma_a_aa, v2rhosigma_a_ab, v2rhosigma_a_bb,
                                             v2rhosigma_b_aa, v2rhosigma_b_ab, v2rhosigma_b_bb].T
                    + v2sigma2
                        + `None` for LDA
                        + `(n_grids, 6)` -> [v2sigma2_aa_aa, v2sigma2_aa_ab, v2sigma2_aa_bb,
                                             v2sigma2_ab_ab, v2sigma2_ab_bb, v2sigma2_bb_bb].T
                    + v2lapl2
                        + `None` for LDA and GGA
                        + `(n_grids, 3)` -> [v2lapl2_aa, v2lapl2_ab, v2lapl2_bb].T
                    + v2tau2
                        + `None` for LDA and GGA
                        + `(n_grids, 3)` -> [v2tau2_aa, v2tau2_ab, v2tau2_bb].T
                    + v2rholapl
                        + `None` for LDA and GGA
                        + `(n_grids, 4)` -> [v2rholapl_a_a, v2rholapl_a_b,
                                             v2rholapl_b_a, v2rholapl_b_b].T
                    + v2rhotau
                        + `None` for LDA and GGA
                        + `(n_grids, 4)` -> [v2rhotau_a_a, v2rhotau_a_b,
                                             v2rhotau_b_a, v2rhotau_b_b].T
                    + v2lapltau
                        + `None` for LDA and GGA
                        + `(n_grids, 4)` -> [v2lapltau_a_a, v2lapltau_a_b,
                                             v2lapltau_b_a, v2lapltau_b_b].T
                    + v2sigmalapl
                        + `None` for LDA and GGA
                        + `(n_grids, 6)` -> [v2sigmalapl_aa_a, v2sigmalapl_aa_b,
                                             v2sigmalapl_ab_a, v2sigmalapl_ab_b,
                                             v2sigmalapl_bb_a, v2sigmalapl_bb_b].T
                    + v2sigmatau
                        + `None` for LDA and GGA
                        + `(n_grids, 6)` -> [v2sigmatau_aa_a, v2sigmatau_aa_b,
                                             v2sigmatau_ab_a, v2sigmatau_ab_b,
                                             v2sigmatau_bb_a, v2sigmatau_bb_b].T
            + kxc
                + `kxc = (v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3, v3rho2lapl,
                          v3rho2tau, v3rhosigmalapl, v3rhosigmatau, v3rholapl2,
                          v3rholapltau, v3rhotau2, v3sigma2lapl, v3sigma2tau,
                          v3sigmalapl2, v3sigmalapltau, v3sigmatau2, v3lapl3,
                          v3lapl2tau, v3lapltau2, v3tau3)`
                + `None` for deriv < 3
                + for R- case, `(n_grids,)` for each of the above
                + for U- case:
                    + v3rho3 # TODO

        """

        ex, vx, fx, kx = self.eval_x(rho, spin, relativity, deriv, omega, verbose)
        ec, vc, fc, kc = self.eval_c(rho, spin, relativity, deriv, omega, verbose)

        exc = self.safe_sum(ex, ec)
        vxc = self.safe_sum(vx, vc)
        fxc = self.safe_sum(fx, fc)
        kxc = self.safe_sum(kx, kc)

        return exc, vxc, fxc, kxc

    def eval_x(
        self,
        rho,
        spin,
        relativity=0,
        deriv=1,
        omega=None,
        verbose=None,
    ):
        match self.x_mode:
            case self.FLAG_ORIGIN:
                return dft.libxc.eval_xc(
                    self.xc_code.x, rho, spin, relativity, deriv, omega, verbose
                )
            case self.FLAG_MODIFY_RHO | self.FLAG_MODIFY_RHO2:
                return self.eval_x_modify_rho(
                    rho, spin, relativity, deriv, omega, verbose
                )
            case _:
                raise NotImplementedError(f"X mode '{self.x_mode}' is not implemented.")

    def eval_x_modify_rho(
        self,
        rho,
        spin,
        relativity=0,
        deriv=1,
        omega=None,
        verbose=None,
    ):
        if spin == self.SINGLET_SPIN:
            ex, vx, fx, kx = self.eval_x_modify_rho_s(
                rho, relativity, deriv, omega, verbose
            )
        else:
            exa, vxa, fxa, kxa = self.eval_x_modify_rho_s(
                self.U_FACTOR_RHO * rho[0], relativity, deriv, omega, verbose
            )
            exb, vxb, fxb, kxb = self.eval_x_modify_rho_s(
                self.U_FACTOR_RHO * rho[1], relativity, deriv, omega, verbose
            )

            rhoa = rho[0][0]
            rhob = rho[1][0]
            ex = (rhoa * exa + rhob * exb) / (rhoa + rhob)

            vx = self.all_concatenate(vxa, vxb, self.LENS_VXC_U, self.U_FACTORS_VXC)
            fx = self.all_concatenate(fxa, fxb, self.LENS_FXC_U, self.U_FACTORS_FXC)
            kx = self.all_concatenate(kxa, kxb, self.LENS_KXC_U, self.U_FACTORS_KXC)

        return ex, vx, fx, kx

    def eval_x_modify_rho_s(self, rho, relativity, deriv, omega, verbose):
        match (self.xc_code.level, self.x_mode):
            case ("LDA", _):
                ex, vx, fx, kx = self.eval_x_modify_rho_s_lda(
                    rho, relativity, deriv, omega, verbose
                )
            case ("GGA", self.FLAG_MODIFY_RHO):
                ex, vx, fx, kx = self.eval_x_modify_rho_s_gga(
                    rho, relativity, deriv, omega, verbose
                )
            case ("MGGA", self.FLAG_MODIFY_RHO):
                ex, vx, fx, kx = self.eval_x_modify_rho_s_mgga(
                    rho, relativity, deriv, omega, verbose
                )
            case ("GGA", self.FLAG_MODIFY_RHO2):
                ex, vx, fx, kx = self.eval_x_modify_rho2_s_gga(
                    rho, relativity, deriv, omega, verbose
                )
            case ("MGGA", self.FLAG_MODIFY_RHO2):
                ex, vx, fx, kx = self.eval_x_modify_rho2_s_mgga(
                    rho, relativity, deriv, omega, verbose
                )
            case _:
                raise NotImplementedError(
                    f"XC level '{self.xc_code.level}' and "
                    f"X mode '{self.x_mode}' combination is not implemented."
                )

        return self.sanitize_value((ex, vx, fx, kx), copy=False)

    def eval_x_modify_rho_s_lda(self, rho, relativity, deriv, omega, verbose):
        rho_aux = rho ** (self.alpha * 3.0 / 4.0)
        ex_aux, vx_aux, fx_aux, _ = dft.libxc.eval_xc(
            self.xc_code.x,
            rho_aux,
            self.SINGLET_SPIN,
            relativity,
            deriv,
            omega,
            verbose,
        )
        factor = rho ** (self.alpha * 3.0 / 4.0 - 1)
        ex = self.Cx * factor * ex_aux

        if deriv > 0:
            vrho_aux = vx_aux[0]
            vrho = self.Cx * self.alpha * 3.0 / 4.0 * factor * vrho_aux
            vx = self.normalize_tuple_size((vrho,), self.LEN_VXC)
            fx = self.normalize_tuple_size((None,), self.LEN_FXC)
        if deriv > 1:
            v2rho2_aux = fx_aux[0]
            v2rho2 = (
                self.Cx * (self.alpha * 3.0 / 4.0 * factor) ** 2 * v2rho2_aux
                + self.Cx
                * (3.0 * self.alpha / 4.0)
                * (3.0 * self.alpha / 4.0 - 1)
                * rho ** (self.alpha * 3.0 / 4.0 - 2)
                * vrho_aux
            )
            fx = self.normalize_tuple_size((v2rho2,), self.LEN_FXC)
        kx = self.normalize_tuple_size((None,), self.LEN_KXC)

        return ex, vx, fx, kx

    def eval_x_modify_rho_s_gga(self, rho, relativity, deriv, omega, verbose):
        ex_aux, vx_aux, fx_aux, _ = dft.libxc.eval_xc(
            self.xc_code.x, rho, self.SINGLET_SPIN, relativity, deriv, omega, verbose
        )
        rho0 = rho[0]
        factor = self.Cx * rho0 ** (self.alpha - 4.0 / 3.0)
        ex = factor * ex_aux
        if deriv > 0:
            vrho_aux, vsigma_aux = vx_aux
            vrho = factor * (vrho_aux + (self.alpha - 4.0 / 3.0) * ex_aux)
            vsigma = factor * vsigma_aux
            vx = self.normalize_tuple_size((vrho, vsigma), self.LEN_VXC)
            fx = self.normalize_tuple_size((None,), self.LEN_FXC)
        if deriv > 1:
            v2rho2_aux, v2rhosigma_aux, v2sigma2_aux = fx_aux
            v2rho2 = (
                self.Cx
                * (self.alpha - 4.0 / 3.0)
                * (self.alpha - 7.0 / 3.0)
                * rho0 ** (self.alpha - 7.0 / 3.0)
                * ex_aux
                + self.Cx
                * 2
                * (self.alpha - 4.0 / 3.0)
                * rho0 ** (self.alpha - 7.0 / 3.0)
                * vrho_aux
                + self.Cx * rho0 ** (self.alpha - 4.0 / 3.0) * v2rho2_aux
            )
            v2rhosigma = (
                self.Cx
                * (self.alpha - 4.0 / 3.0)
                * rho0 ** (self.alpha - 7.0 / 3.0)
                * vsigma_aux
                + self.Cx * rho0 ** (self.alpha - 4.0 / 3.0) * v2rhosigma_aux
            )
            v2sigma2 = self.Cx * rho0 ** (self.alpha - 4.0 / 3.0) * v2sigma2_aux
            fx = self.normalize_tuple_size((v2rho2, v2rhosigma, v2sigma2), self.LEN_FXC)
        kx = self.normalize_tuple_size((None,), self.LEN_KXC)

        return ex, vx, fx, kx

    def eval_x_modify_rho_s_mgga(self, rho, relativity, deriv, omega, verbose):
        ex_aux, vx_aux, fx_aux, _ = dft.libxc.eval_xc(
            self.xc_code.x, rho, self.SINGLET_SPIN, relativity, deriv, omega, verbose
        )
        rho0 = rho[0]
        factor = self.Cx * rho0 ** (self.alpha - 4.0 / 3.0)
        ex = factor * ex_aux
        if deriv > 0:
            vrho_aux, vsigma_aux, _, vtau_aux = vx_aux
            vrho = factor * (vrho_aux + (self.alpha - 4.0 / 3.0) * ex_aux)
            vsigma = factor * vsigma_aux
            vtau = factor * vtau_aux
            vx = self.normalize_tuple_size((vrho, vsigma, None, vtau), self.LEN_VXC)
            fx = self.normalize_tuple_size((None,), self.LEN_FXC)
        if deriv > 1:
            v2rho2_aux, v2rhosigma_aux, v2sigma2_aux = fx_aux
            v2rho2 = (
                self.Cx
                * (self.alpha - 4.0 / 3.0)
                * (self.alpha - 7.0 / 3.0)
                * rho0 ** (self.alpha - 7.0 / 3.0)
                * ex_aux
                + self.Cx
                * 2
                * (self.alpha - 4.0 / 3.0)
                * rho0 ** (self.alpha - 7.0 / 3.0)
                * vrho_aux
                + self.Cx * rho0 ** (self.alpha - 4.0 / 3.0) * v2rho2_aux
            )
            v2rhosigma = (
                self.Cx
                * (self.alpha - 4.0 / 3.0)
                * rho0 ** (self.alpha - 7.0 / 3.0)
                * vsigma_aux
                + self.Cx * rho0 ** (self.alpha - 4.0 / 3.0) * v2rhosigma_aux
            )
            v2sigma2 = self.Cx * rho0 ** (self.alpha - 4.0 / 3.0) * v2sigma2_aux
            fx = self.normalize_tuple_size((v2rho2, v2rhosigma, v2sigma2), self.LEN_FXC)
        kx = self.normalize_tuple_size((None,), self.LEN_KXC)

        return ex, vx, fx, kx

    def eval_x_modify_rho2_s_gga(self, rho, relativity, deriv, omega, verbose):
        rho0 = rho[0]
        rho_aux = rho.copy()
        rho_aux[0] = rho0 ** (self.alpha * 3.0 / 4.0)
        ex_aux, vx_aux, fx_aux, _ = dft.libxc.eval_xc(
            self.xc_code.x,
            rho_aux,
            self.SINGLET_SPIN,
            relativity,
            deriv,
            omega,
            verbose,
        )
        factor = rho0 ** (self.alpha * 3.0 / 4.0 - 1)
        ex = self.Cx * factor * ex_aux

        if deriv > 0:
            vrho_aux, vsigma_aux = vx_aux
            vrho = self.Cx * self.alpha * 3.0 / 4.0 * factor * vrho_aux
            vsigma = self.Cx * vsigma_aux
            vx = self.normalize_tuple_size((vrho, vsigma), self.LEN_VXC)
            fx = self.normalize_tuple_size((None,), self.LEN_FXC)
        if deriv > 1:
            v2rho2_aux, v2rhosigma_aux, v2sigma2_aux = fx_aux
            v2rho2 = (
                self.Cx * (self.alpha * 3.0 / 4.0 * factor) ** 2 * v2rho2_aux
                + self.Cx
                * (self.alpha * 3.0 / 4.0)
                * (self.alpha * 3.0 / 4.0 - 1)
                * factor
                / rho0
                * vrho_aux
            )
            v2rhosigma = self.Cx * self.alpha * 3.0 / 4.0 * factor * v2rhosigma_aux
            v2sigma2 = self.Cx * v2sigma2_aux
            fx = self.normalize_tuple_size(
                (
                    v2rho2,
                    v2rhosigma,
                    v2sigma2,
                ),
                self.LEN_FXC,
            )
        kx = self.normalize_tuple_size((None,), self.LEN_KXC)

        return ex, vx, fx, kx

    def eval_x_modify_rho2_s_mgga(self, rho, relativity, deriv, omega, verbose):
        rho0 = rho[0]
        rho_aux = rho.copy()
        rho_aux[0] = rho0 ** (self.alpha * 3.0 / 4.0)
        ex_aux, vx_aux, fx_aux, _ = dft.libxc.eval_xc(
            self.xc_code.x,
            rho_aux,
            self.SINGLET_SPIN,
            relativity,
            deriv,
            omega,
            verbose,
        )
        factor = rho0 ** (self.alpha * 3.0 / 4.0 - 1)
        ex = self.Cx * factor * ex_aux

        if deriv > 0:
            vrho_aux, vsigma_aux, _, vtau_aux = vx_aux  # vrho, vsigma, vlapl, vtau
            vrho = self.Cx * self.alpha * 3.0 / 4.0 * factor * vrho_aux
            vsigma = self.Cx * vsigma_aux
            vtau = self.Cx * vtau_aux
            vx = self.normalize_tuple_size((vrho, vsigma, None, vtau), self.LEN_VXC)
            fx = self.normalize_tuple_size((None,), self.LEN_FXC)
        if deriv > 1:
            # (v2rho2, v2rhosigma, v2sigma2, v2lapl2, v2tau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)
            (
                v2rho2_aux,
                v2rhosigma_aux,
                v2sigma2_aux,
                _,
                v2tau2_aux,
                _,
                v2rhotau_aux,
                _,
                _,
                v2sigmatau_aux,
            ) = fx_aux
            v2rho2 = (
                self.Cx * (self.alpha * 3.0 / 4.0 * factor) ** 2 * v2rho2_aux
                + self.Cx
                * (3.0 * self.alpha / 4.0)
                * (3.0 * self.alpha / 4.0 - 1)
                * rho0 ** (self.alpha * 3.0 / 4.0 - 2)
                * vrho_aux
            )
            v2rhosigma = self.Cx * self.alpha * 3.0 / 4.0 * factor * v2rhosigma_aux
            v2rhotau = self.Cx * self.alpha * 3.0 / 4.0 * factor * v2rhotau_aux
            v2sigma2 = self.Cx * v2sigma2_aux
            v2sigmatau = self.Cx * v2sigmatau_aux
            v2tau2 = self.Cx * v2tau2_aux
            fx = self.normalize_tuple_size(
                (
                    v2rho2,
                    v2rhosigma,
                    v2sigma2,
                    None,
                    v2tau2,
                    None,
                    v2rhotau,
                    None,
                    None,
                    v2sigmatau,
                ),
                self.LEN_FXC,
            )
        kx = self.normalize_tuple_size((None,), self.LEN_KXC)

        return ex, vx, fx, kx

    def eval_c(
        self,
        rho,
        spin,
        relativity=0,
        deriv=1,
        omega=None,
        verbose=None,
    ):
        match self.c_mode:
            case self.FLAG_ORIGIN:
                return dft.libxc.eval_xc(
                    self.xc_code.c, rho, spin, relativity, deriv, omega, verbose
                )
            case _:
                raise NotImplementedError(
                    f"Correlation functional mode '{self.c_mode}' is not implemented."
                )


def MKS(mol):
    if mol.spin == 0:
        return MRKS(mol)
    else:
        return MUKS(mol)


class MRKS(dft.rks.RKS):
    def __init__(self, mol, **kwargs):
        super().__init__(mol, **kwargs)

    @property
    def mxc(self):
        if not hasattr(self, "_mxc"):
            raise AttributeError("xc is not set.")
        return self._mxc

    @mxc.setter
    def mxc(self, value: str):
        """Set the exchange-correlation functional.

        Args:
            value (str): 字符串形式的泛函名，格式为"base_xc[_alpha-<alpha>][_Cx-<Cx>]..."
                例如："PBE_alpha-1.4_Cx-1.0" 或 "acPBE_alpha-1.4_Cx-1.0"
                这里，下划线`_`用于分隔不同的参数，减号`-`用于分隔参数名和参数值。
        """
        args = value.split("_")
        builtin_xc = args[0]

        mxc = ModifiedXC.builtin(builtin_xc)

        for xc_arg in args[1:]:
            key, val = xc_arg.split("-")
            if hasattr(mxc, key):
                setattr(mxc, key, float(val))
            else:
                raise ValueError(
                    f"Invalid parameter '{key}' for ModifiedXC({builtin_xc})."
                )

        xc_level = mxc.xc_code.level
        self.define_xc_(mxc.eval_xc, xc_level)
        self._mxc = mxc


class MUKS(dft.uks.UKS):
    def __init__(self, mol, **kwargs):
        super().__init__(mol, **kwargs)

    @property
    def mxc(self):
        if not hasattr(self, "_mxc"):
            raise AttributeError("xc is not set.")
        return self._mxc

    @mxc.setter
    def mxc(self, value: str):
        """Set the exchange-correlation functional.

        Args:
            value (str): 字符串形式的泛函名，格式为"base_xc[_alpha-<alpha>][_Cx-<Cx>]..."
                例如："PBE_alpha-1.4_Cx-1.0" 或 "acPBE_alpha-1.4_Cx-1.0"
                这里，下划线`_`用于分隔不同的参数，减号`-`用于分隔参数名和参数值。
        """
        args = value.split("_")
        builtin_xc = args[0]

        mxc = ModifiedXC.builtin(builtin_xc)

        for xc_arg in args[1:]:
            key, val = xc_arg.split("-")
            setattr(mxc, key, float(val))

        xc_level = mxc.xc_code.level
        self.define_xc_(mxc.eval_xc, xc_level)
        self._mxc = mxc


if __name__ == "__main__":
    from pyscf import dft, gto  # type: ignore

    mol = gto.M(
        atom="H 0 0 0",
        charge=-1,
        spin=0,
        basis="def2-svp",
    )

    # 一个直接的方式
    mxc1 = ModifiedXC(
        xc_code="PBE", x_mode="modify-rho", c_mode="origin", alpha=1.4, Cx=1.0
    )
    ks1 = dft.KS(mol)
    ks1.define_xc_(mxc1.eval_xc, "GGA")
    ks1.kernel()

    # 一个简便的方式
    mxc2 = ModifiedXC.builtin("acPBE1")
    mxc2.alpha = 1.4
    mxc2.Cx = 1.0
    ks2 = dft.KS(mol)
    ks2.define_xc_(mxc2.eval_xc, "GGA")
    ks2.kernel()

    # 一个更简便的方法
    ks3 = MKS(mol)
    ks3.mxc = "acPBE1_alpha-1.4_Cx-1.0"
    ks3.kernel()