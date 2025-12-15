from myplugins.Geometry import QUEST, Adia7

import os
import numpy as np
import inspect

class GaussianJobGenerator:
    """
    Gaussian任务文件生成器
    用于将量子化学数据集转换为Gaussian输入文件
    """
    
    def __init__(self, memory='8GB', nprocshared=8, chk=True):
        """
        初始化生成器
        
        Parameters:
        -----------
        memory : str
            内存大小，默认'8GB'
        nprocshared : int
            使用的CPU核心数，默认8
        chk : bool
            是否生成chk文件，默认True
        """
        self.memory = memory
        self.nprocshared = nprocshared
        self.chk = chk
        
        # Bohr到Angstrom的转换因子
        self.bohr_to_angstrom = 0.52917721067
    
    def _get_caller_directory(self):
        """
        获取调用此函数的脚本所在目录
        
        Returns:
        --------
        str
            调用脚本的目录路径
        """
        # 获取调用栈帧
        frame = inspect.currentframe()
        try:
            # 回溯到调用此函数的帧
            caller_frame = frame.f_back.f_back  # 跳过当前方法和调用方法
            caller_file = caller_frame.f_globals.get('__file__', None)
            if caller_file:
                return os.path.dirname(os.path.abspath(caller_file))
            else:
                # 如果在交互式环境中运行，使用当前工作目录
                return os.getcwd()
        finally:
            del frame  # 避免引用循环
    
    def convert_geometry_to_angstrom(self, geom_string, current_unit='Bohr'):
        """
        将分子坐标转换为埃单位
        
        Parameters:
        -----------
        geom_string : str
            分子几何结构的字符串
        current_unit : str
            当前单位，'Bohr'或'A'
            
        Returns:
        --------
        str
            转换后的几何结构字符串
        """
        lines = geom_string.strip().split('\n')
        converted_lines = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                element = parts[0]
                try:
                    # 转换坐标
                    if current_unit == 'Bohr':
                        x = float(parts[1]) * self.bohr_to_angstrom
                        y = float(parts[2]) * self.bohr_to_angstrom
                        z = float(parts[3]) * self.bohr_to_angstrom
                    else:  # 已经是Angstrom
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                    
                    # 格式化输出，保留6位小数
                    converted_line = f"{element:>2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}"
                    converted_lines.append(converted_line)
                except ValueError:
                    # 如果转换失败，保持原样
                    converted_lines.append(line.strip())
        
        return '\n'.join(converted_lines)
    
    def generate_gaussian_input(self, geometry_class, route_line, output_dir='gaussian_jobs'):
        """
        为几何数据集生成Gaussian输入文件
        
        Parameters:
        -----------
        geometry_class : GeometryBase subclass
            几何数据集类的实例
        route_line : str
            Gaussian任务行（如：'# B3LYP/6-31G(d) Opt Freq'）
        output_dir : str
            输出目录名称，默认'gaussian_jobs'
        """
        # 获取调用脚本的目录
        base_dir = self._get_caller_directory()
        full_output_dir = os.path.join(base_dir, output_dir)
        
        # 创建输出目录
        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir)
        
        # 获取数据集中的所有分子名称
        names = geometry_class.get_names()
        
        for name in names:
            # 获取分子信息
            geom = geometry_class.get_geom(name)
            charge = geometry_class.get_charge(name)
            spin_2s = geometry_class.get_spin(name)  # 这是2S
            multiplicity = spin_2s + 1  # Gaussian中使用2S+1
            
            # 转换坐标单位
            converted_geom = self.convert_geometry_to_angstrom(geom, geometry_class.unit)
            
            # 生成文件名
            filename = f"{name}.gjf"
            filepath = os.path.join(full_output_dir, filename)
            
            # 写入Gaussian输入文件
            with open(filepath, 'w') as f:
                # 写入抬头信息
                if self.chk:
                    f.write(f"%chk={name}.chk\n")
                f.write(f"%mem={self.memory}\n")
                f.write(f"%nprocshared={self.nprocshared}\n")
                
                # 写入任务行
                f.write(f"{route_line}\n")
                f.write("\n")
                f.write(f"{name}\n")
                f.write("\n")
                
                # 写入电荷和自旋多重度
                f.write(f"{charge} {multiplicity}\n")
                
                # 写入几何结构
                f.write(converted_geom)
                f.write("\n")
                f.write("\n")  # 空行结束
            
            print(f"生成文件: {filepath}")
    
    def generate_gaussian_input_single(self, geometry_class, molecule_name, route_line, 
                                     output_dir='gaussian_jobs', custom_charge=None, custom_spin=None):
        """
        为单个分子生成Gaussian输入文件
        
        Parameters:
        -----------
        geometry_class : GeometryBase subclass
            几何数据集类的实例
        molecule_name : str
            分子名称
        route_line : str
            Gaussian任务行
        output_dir : str
            输出目录名称
        custom_charge : int, optional
            自定义电荷，如果不提供则使用数据集中的默认值
        custom_spin : int, optional
            自定义自旋(2S)，如果不提供则使用数据集中的默认值
        """
        # 获取调用脚本的目录
        base_dir = self._get_caller_directory()
        full_output_dir = os.path.join(base_dir, output_dir)
        
        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir)
        
        # 获取分子信息
        geom = geometry_class.get_geom(molecule_name)
        charge = custom_charge if custom_charge is not None else geometry_class.get_charge(molecule_name)
        spin_2s = custom_spin if custom_spin is not None else geometry_class.get_spin(molecule_name)
        multiplicity = spin_2s + 1  # Gaussian中使用2S+1
        
        # 转换坐标单位
        converted_geom = self.convert_geometry_to_angstrom(geom, geometry_class.unit)
        
        # 生成文件名
        filename = f"{molecule_name}.gjf"
        filepath = os.path.join(full_output_dir, filename)
        
        # 写入Gaussian输入文件
        with open(filepath, 'w') as f:
            # 写入抬头信息
            if self.chk:
                f.write(f"%chk={molecule_name}.chk\n")
            f.write(f"%mem={self.memory}\n")
            f.write(f"%nprocshared={self.nprocshared}\n")
            
            # 写入任务行
            f.write(f"{route_line}\n")
            f.write("\n")
            f.write(f"{molecule_name}\n")
            f.write("\n")
            
            # 写入电荷和自旋多重度
            f.write(f"{charge} {multiplicity}\n")
            
            # 写入几何结构
            f.write(converted_geom)
            f.write("\n")
            f.write("\n")  # 空行结束
        
        print(f"生成文件: {filepath}")

# 使用示例函数
def generate_quest_gaussian_jobs(route_line='# B3LYP/6-31G(d) Opt Freq', 
                               memory='8GB', 
                               nprocshared=8,
                               output_dir='gaussian_jobs',
                               dataset_class=None):
    """
    便捷函数：为QUEST数据集生成Gaussian任务文件
    
    Parameters:
    -----------
    route_line : str
        Gaussian任务行
    memory : str
        内存大小
    nprocshared : int
        CPU核心数
    output_dir : str
        输出目录
    dataset_class : GeometryBase subclass
        数据集类，如果为None则使用QUEST
    """
    if dataset_class is None:
        dataset_class = QUEST()
    
    # 获取调用脚本的目录
    frame = inspect.currentframe()
    try:
        caller_frame = frame.f_back
        caller_file = caller_frame.f_globals.get('__file__', None)
        if caller_file:
            base_dir = os.path.dirname(os.path.abspath(caller_file))
        else:
            base_dir = os.getcwd()
    finally:
        del frame
    
    generator = GaussianJobGenerator(memory=memory, nprocshared=nprocshared)
    
    # 修改输出目录为基于调用脚本的完整路径
    full_output_dir = os.path.join(base_dir, output_dir)
    
    # 创建输出目录
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
    
    # 获取数据集中的所有分子名称
    names = dataset_class.get_names()
    
    for name in names:
        # 获取分子信息
        geom = dataset_class.get_geom(name)
        charge = dataset_class.get_charge(name)
        spin_2s = dataset_class.get_spin(name)  # 这是2S
        multiplicity = spin_2s + 1  # Gaussian中使用2S+1
        
        # 转换坐标单位
        converted_geom = generator.convert_geometry_to_angstrom(geom, dataset_class.unit)
        
        # 生成文件名
        filename = f"{name}.gjf"
        filepath = os.path.join(full_output_dir, filename)
        
        # 写入Gaussian输入文件
        with open(filepath, 'w') as f:
            # 写入抬头信息
            if generator.chk:
                f.write(f"%chk={name}.chk\n")
            f.write(f"%mem={memory}\n")
            f.write(f"%nprocshared={nprocshared}\n")
            
            # 写入任务行
            f.write(f"{route_line}\n")
            f.write("\n")
            f.write(f"{name}\n")
            f.write("\n")
            
            # 写入电荷和自旋多重度
            f.write(f"{charge} {multiplicity}\n")
            
            # 写入几何结构
            f.write(converted_geom)
            f.write("\n")
            f.write("\n")  # 空行结束
        
        print(f"生成文件: {filepath}")

# 如果直接运行此脚本，提供一个简单的测试
if __name__ == "__main__":
    # 测试代码
    generator = GaussianJobGenerator()
    
    # 为QUEST1数据集生成测试文件
    quest1 = QUEST.QUEST1()
    quest8 = QUEST.QUEST8()
    adia7 = Adia7.Adia7()
    route = "# HF/6-31G Opt"
    
    print("生成QUEST1测试文件...")
    generator.generate_gaussian_input(quest1, route, "test_gaussian_jobs_quest1")
    print("生成QUEST8测试文件...")
    generator.generate_gaussian_input(quest8, route, "test_gaussian_jobs_quest8")
    print("生成Adia7测试文件...")
    generator.generate_gaussian_input(adia7, route, "test_gaussian_jobs_adia7")

