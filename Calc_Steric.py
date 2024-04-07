from atomic_data import PeriodicTable
import numpy as np
import matplotlib.pyplot as plt

class Calc_Steric():
    def __init__(self,mol_file,x_pix=200,y_pix=200):
        self.mol_file = mol_file
        self.x_pix = x_pix
        self.y_pix = y_pix
        if self.mol_file.endswith('.mol2'):
            # 读取mol2文件，返回原子列表和坐标列表
            self.atomlist = self.read_mol2()
            self.calc_masscenter()
            self.calc_intersection_points()
        else :
            print('不支持该文件格式')
            return 1
    def read_mol2(self):
        with open(self.mol_file, 'r') as f:
            lines = f.readlines()
        atoms = []
        atom_weights = []
        atom_vdwrs = []
        atom_lables = []
        atom_positions = []
        for line in lines:
            if line.startswith('@<TRIPOS>ATOM'):
                firstatom = lines.index(line)+1
            elif line.startswith('@<TRIPOS>BOND'):
                firstbond = lines.index(line)+1
        atom_lines = lines[firstatom:firstbond-1]
        # print(atom_lines)
        for atom_line in atom_lines:
            atom= atom_line.split()[5]
            atoms.append(atom)
            atom_weights.append(PeriodicTable().mass(atom))
            atom_vdwrs.append(PeriodicTable().vdw_radius(atom))
            atom_lables.append(atom_line.split()[1])
            n = [float(x) for x in atom_line.split()[2:5]]
            n = np.array(n)
            atom_positions.append(n)
            # print(PeriodicTable().mass(atom))
        # print(atom_positions)
        self.atom_weights = atom_weights
        self.atom_vdwrs = np.array(atom_vdwrs)
        volume = 4/3*np.pi*pow(self.atom_vdwrs,3)
        self.volume = volume
        self.atom_lables = atom_lables
        self.atom_positions = np.array(atom_positions)
        # print(self.atom_positions)

    def calc_masscenter(self):
        # 计算分子重心坐标
        atom_weights = np.array([self.atom_weights,self.atom_weights,self. atom_weights]).T  #.T 表示转置  
        # print(self.atom_vdwrs) 
    def _polar2xyz(self,r,theta,phi):
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        return np.array([x,y,z])
    
    def _xyz2polar(self,x,y,z):
        r = np.sqrt(x**2+y**2+z**2)
        theta = np.arccos(z/r)
        phi = np.arctan2(y,x)
        return np.array([r,theta,phi])

    def calc_intersection_points(self):
        # 随机从球坐标空间取方向
        phis = np.linspace(0, 2*np.pi, self.x_pix)
        thetas = np.linspace(0, np.pi, self.y_pix)
        θ, φ = np.meshgrid(thetas, phis)
        obs_index = 0
        for observer_pos in self.atom_positions:
            maps = []   
            for phi in phis:
                for theta in thetas:
                    # 最大距离
                    max_d = 0
                    # 计算方向向量
                    direction_vector = np.array([
                        np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)])
                    # 球体方程：(x - cx)^2 + (y - cy)^2 + (z - cz)^2 = r^2
                    # 将球体方程转换为标准形式：x^2 + y^2 + z^2 - 2cx - 2cy - 2cz + c^2 - r^2 = 0
                    a = direction_vector[0]**2 + direction_vector[1]**2 + direction_vector[2]**2
                    atom_index = 0
                    for atom_pos in self.atom_positions:
                        # print(atom_pos)
                        b = 2 * ((observer_pos[0] - atom_pos[0]) * direction_vector[0] + 
                                (observer_pos[1] - atom_pos[1]) * direction_vector[1] +
                                (observer_pos[2] - atom_pos[2]) * direction_vector[2])
                        c = atom_pos[0]**2 + atom_pos[1]**2 + atom_pos[2]**2 - self.atom_vdwrs[atom_index]**2
                        # 计算根
                        discriminant = b**2 - 4 * a * c
                        if discriminant >= 0:
                            root1 = (-b + np.sqrt(discriminant)) / (2 * a)
                            root2 = (-b - np.sqrt(discriminant)) / (2 * a)
                            # 计算两个根的坐标
                            intersection_point1 = observer_pos + root1 * direction_vector
                            intersection_point2 = observer_pos + root2 * direction_vector
                            # 计算交点与观察点的距离
                            d1 = np.sqrt((intersection_point1[0] - observer_pos[0])**2 +
                                        (intersection_point1[1] - observer_pos[1])**2 +
                                        (intersection_point1[2] - observer_pos[2])**2)
                            d2 = np.sqrt((intersection_point2[0] - observer_pos[0])**2 +
                                        (intersection_point2[1] - observer_pos[1])**2 +
                                        (intersection_point2[2] - observer_pos[2])**2)
                            max_d = max(max_d, d1, d2)
                        atom_index = atom_index + 1
                    if max_d <= self.atom_vdwrs[obs_index]+0.1:
                        max_d = 0
                        print('自身交点')
                    map = np.array(max_d)
                    maps.append(map)
            maps = np.array(maps).reshape(self.x_pix, self.y_pix)
            # print(maps.shape)
            plt.figure(figsize=(10, 5))
            plt.pcolormesh(φ,θ,maps, cmap='RdBu') # 设置颜色映射
            plt.colorbar(label='d(Å)') # 添加颜色条
            plt.xticks(np.linspace(0, 2*np.pi, 5))
            plt.yticks(np.linspace(0, np.pi, 3))
            plt.xlabel('φ')
            plt.ylabel('θ')
            plt.title(self.atom_lables[obs_index])
            # plt.show()
            plt.savefig(f'./picture/{self.atom_lables[obs_index]}.png')
            plt.close()
            obs_index = obs_index + 1
            

          
# 测试用
Calc_Steric('x.mol2',200,100)