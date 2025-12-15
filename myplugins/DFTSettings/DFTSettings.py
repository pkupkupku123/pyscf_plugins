import shutil


class DFT_Settings():
    """
    A class to store the DFT settings for a calculation.

    Attributes
    ----------
    functional : str
        The functional to be used in the DFT calculation.
    basis : str
        The basis set to be used in the calculation.
    verbose : int
        The verbosity level of the output.
    losc_correction : str
        The type of LOSC correction to be used.
    energy_window : list (with two floats, [min, max], units in eV)
        The energy window to be used in the LOSC localization.
    max_iter : int
        The maximum number of iterations in the LOSC localization.
    curvature_version : int
        The version of the curvature to be used in the LOSC procedure.
    task_tag : str
        The tag for the task.
    output_name : str
        The name of the output file.

    Methods
    -------
    __init__(file_name)
        Initialize the DFT settings from a file.

    read_dft_settings(file_name)
        Read the DFT settings from a file and update the attributes.
    """
    def __init__(self, file_name):
        self.functional =   'LDA'
        self.basis      =   'cc-pvdz'
        self.fitbasis   =   'cc-pvdz-ri'
        self.verbose    =   3
        self.losc_correction    =    'LDA'
        self.energy_window      =   [-30, 10]
        self.max_iter           =   1000
        self.gamma              =   0.707
        self.zeta               =   8.0
        self.curvature_version  =   2
        self.data_window        =   [None, None]
        self.max_memory         =   4000

        self.task_tag           =   '20000626T001'
        self.output_name        =   'output.csv'
        self.read_dft_settings(file_name)
        shutil.copy(file_name, 'dft_settings_' + self.task_tag + '.txt')
        pass

    def read_dft_settings(self, file_name):
        file    =   open(file_name, "r")
        lines   =   file.read().split("\n")

        for line in lines:
            if line.startswith("functional"):
                self.functional =   line.split()[1]
            elif line.startswith("basis"):
                self.basis      =   line.split()[1]
            elif line.startswith("fitbasis"):
                self.fitbasis   =   line.split()[1]
            elif line.startswith("verbose"):
                self.verbose    =   int(line.split()[1])
            elif line.startswith("losc_correction"):
                self.losc_correction    =   line.split()[1]
            elif line.startswith("energy_window"):
                self.energy_window      =   [float(line.split()[1]), float(line.split()[2])]
            elif line.startswith("max_iter"):
                self.max_iter           =   int(line.split()[1])
            elif line.startswith("curvature_version"):
                self.curvature_version  =   int(line.split()[1])
            elif line.startswith("gamma"):
                self.gamma              =   float(line.split()[1])
            elif line.startswith("zeta"):
                self.zeta               =   float(line.split()[1])
            elif line.startswith("task_tag"):
                self.task_tag           =   line.split()[1]
            elif line.startswith("output_name"):
                self.output_name        =   line.split()[1]
            elif line.startswith("data_window"):
                if line.split()[1] == "None":
                    start_position = None
                else:
                    start_position = int(line.split()[1])
                if line.split()[2] == "None":
                    end_position = None
                else:
                    end_position = int(line.split()[2])
                self.data_window   =   [start_position, end_position]
            elif line.startswith("max_memory"):
                if line.split()[1] != "None":
                    self.max_memory = int(line.split()[1])
        return None

    
    def print_dft_settings(self):
        print("functional: ", self.functional)
        print("basis: ", self.basis)
        print("fitbasis: ", self.fitbasis)
        print("verbose: ", self.verbose)
        print("losc_correction: ", self.losc_correction)
        print("energy_window: ", self.energy_window)
        print("max_iter: ", self.max_iter)
        print("curvature_version: ", self.curvature_version)
        print("localizer_gamma: ", self.gamma)
        print("curvature2_zeta: ", self.zeta)
        print("data_window: ", self.data_window)
        print("task_tag: ", self.task_tag)
        print("output_name: ", self.output_name)
        print("max_memory: ", self.max_memory)
        return None