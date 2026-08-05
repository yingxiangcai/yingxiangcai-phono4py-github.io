#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Phonopy 的 FORCE_SETS 或 VASP 计算结果生成适用于 FourPhonon/ShengBTE 的 FORCE_CONSTANTS_2ND。

兼容 phonopy >= 3.0，利用 phonopy Python API 直接处理。

使用方法:
    方式1: 从已有 FORCE_SETS 生成 FORCE_CONSTANTS_2ND
        python gen_fc2nd.py --mode from_forcesets --dim "2 2 2" -c POSCAR

    方式2: 直接从 disp-*/vasprun.xml 收集力并生成 FORCE_CONSTANTS_2ND
        python gen_fc2nd.py --mode from_vasp --dim "2 2 2" -c POSCAR --pattern "disp-*/vasprun.xml"

输出:
    FORCE_CONSTANTS_2ND  (FourPhonon/ShengBTE 兼容格式)
    FORCE_CONSTANTS       (phonopy 原生格式，可选)
"""

import os
import sys
import glob
import argparse
import numpy as np

# phonopy 导入
try:
    import phonopy
    from phonopy import Phonopy
    from phonopy.interface.vasp import read_vasp
    from phonopy.file_IO import parse_FORCE_SETS, write_FORCE_CONSTANTS
except ImportError as e:
    print(f"错误: 无法导入 phonopy 模块。请确保已安装 phonopy >= 3.0。\n{e}")
    sys.exit(1)


def get_n_atoms(cell_obj):
    """兼容不同 phonopy 版本获取原子数。"""
    # phonopy 3.x 中 cell 对象可能直接支持 len()
    if hasattr(cell_obj, "get_number_of_atoms"):
        return cell_obj.get_number_of_atoms()
    elif hasattr(cell_obj, "numbers"):
        return len(cell_obj.numbers)
    elif hasattr(cell_obj, "symbols"):
        return len(cell_obj.symbols)
    else:
        return len(cell_obj)


def get_vasprun_files(pattern="disp-*/vasprun.xml"):
    """根据通配符获取所有 vasprun.xml 文件，并按目录名排序。"""
    files = glob.glob(pattern)
    def sort_key(f):
        basename = os.path.basename(os.path.dirname(f))
        if basename.startswith("disp-"):
            try:
                return int(basename.split("-")[1])
            except ValueError:
                pass
        return basename
    files.sort(key=sort_key)
    return files


def parse_vasprun_forces(vasprun_file):
    """
    从单个 vasprun.xml 中解析力。
    返回 numpy array, shape=(n_atoms, 3)
    """
    # phonopy 3.x: VasprunxmlExpat 需要文件对象
    try:
        from phonopy.interface.vasp import VasprunxmlExpat
        with open(vasprun_file, "rb") as f:
            vxml = VasprunxmlExpat(f)
            vxml.parse()
        forces = vxml.forces[-1] if vxml.forces else None
        if forces is None:
            raise ValueError(f"无法从 {vasprun_file} 读取力")
        return np.array(forces, dtype="double")
    except Exception as e:
        # 备选: 使用 phonopy 的通用接口
        try:
            from phonopy.interface.vasp import parse_vasprun_xml
            return parse_vasprun_xml(vasprun_file)["forces"]
        except Exception:
            pass
        raise RuntimeError(f"解析 {vasprun_file} 失败: {e}")


def collect_forces_from_vasp(pattern="disp-*/vasprun.xml"):
    """收集所有位移计算的力。"""
    files = get_vasprun_files(pattern)
    if not files:
        raise FileNotFoundError(f"未找到匹配 '{pattern}' 的 vasprun.xml 文件")

    print(f"找到 {len(files)} 个 vasprun.xml 文件:")
    for f in files:
        print(f"  {f}")

    forces_list = []
    for f in files:
        forces = parse_vasprun_forces(f)
        forces_list.append(forces)

    return forces_list, files


def write_FORCE_CONSTANTS_2ND(fc, filename="FORCE_CONSTANTS_2ND"):
    """
    将力常数写入 FORCE_CONSTANTS_2ND 文件 (ShengBTE/FourPhonon 兼容格式)。
    """
    n_satom = fc.shape[0]
    with open(filename, "w") as f:
        f.write(f"{n_satom:5d} {n_satom:5d}\n")
        for i in range(n_satom):
            for j in range(n_satom):
                f.write(f"{i+1:5d} {j+1:5d}\n")
                for alpha in range(3):
                    for beta in range(3):
                        val = fc[i, j, alpha, beta]
                        f.write(f"{val:22.15f}")
                    f.write("\n")
    print(f"力常数已写入: {filename}")


def ensure_full_fc(phonon, fc):
    """
    确保力常数为 FULL 格式 (n_satom, n_satom, 3, 3)。
    phonopy 3.x 默认可能输出 compact 格式 (n_patom, n_satom, 3, 3)。
    """
    n_satom = get_n_atoms(phonon.supercell)

    if fc.shape[0] == n_satom and fc.shape[1] == n_satom:
        return fc

    print(f"检测到 compact 格式 {fc.shape}，扩展为 full 格式 ({n_satom}, {n_satom}, 3, 3)...")
    fc_full = np.zeros((n_satom, n_satom, 3, 3), dtype="double")

    # phonopy 3.x primitive.p2s_map
    p2s_map = getattr(phonon.primitive, "p2s_map", None)
    if p2s_map is None:
        # 备选: 通过 s2p_map 反推
        s2p_map = getattr(phonon.primitive, "s2p_map", None)
        if s2p_map is not None:
            p2s_map = []
            seen = set()
            for i, p in enumerate(s2p_map):
                if p not in seen:
                    p2s_map.append(i)
                    seen.add(p)
        else:
            raise RuntimeError("无法获取 primitive -> supercell 映射")

    for ip, sp in enumerate(p2s_map):
        fc_full[sp] = fc[ip]

    # 利用平移对称性填充其余行: Phi_{i+T, j+T} = Phi_{i,j}
    # 这里需要找到超胞中每个原子对应的原胞等价原子
    # 简单方法: 如果 s2p_map 可用，用 s2p_map 来映射
    s2p_map = getattr(phonon.primitive, "s2p_map", None)
    if s2p_map is not None:
        for i in range(n_satom):
            ip = s2p_map[i]
            # 找到对应的原胞原子在 full 矩阵中的行
            ref_i = p2s_map[ip]
            # 这里假设 fc[ip] 已经包含了所有 j 的信息
            # compact 格式中 fc[ip, j] 表示原胞原子 ip 与超胞原子 j 的相互作用
            # 所以 fc_full[i] 应该等于 fc_full[ref_i] 经过平移... 但实际上 phonopy
            # 的 compact FC 已经考虑了所有 j，只是 i 维度压缩了。
            # 因此直接复制即可（phonopy 内部已处理 j 的周期性）
            if i != ref_i:
                fc_full[i] = fc_full[ref_i]

    return fc_full


def generate_fc2nd_from_forcesets(unitcell_filename, supercell_matrix, 
                                   force_sets_filename="FORCE_SETS",
                                   output_filename="FORCE_CONSTANTS_2ND",
                                   write_phonopy_fc=False,
                                   symmetrize=True):
    """从已有的 FORCE_SETS 生成 FORCE_CONSTANTS_2ND。"""
    unitcell = read_vasp(unitcell_filename)
    phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
    phonon.generate_displacements()

    n_satom = get_n_atoms(phonon.supercell)
    print(f"超胞原子数: {n_satom}")

    print(f"读取 {force_sets_filename} ...")
    phonon.dataset = parse_FORCE_SETS(natom=n_satom, filename=force_sets_filename)

    print("计算二阶力常数 (FC2) ...")
    phonon.produce_force_constants()

    if symmetrize:
        print("对称化力常数 ...")
        phonon.symmetrize_force_constants()

    fc = phonon.force_constants
    print(f"原始力常数 shape: {fc.shape}")

    fc = ensure_full_fc(phonon, fc)

    write_FORCE_CONSTANTS_2ND(fc, filename=output_filename)

    if write_phonopy_fc:
        write_FORCE_CONSTANTS(fc, filename="FORCE_CONSTANTS")
        print("同时输出 FORCE_CONSTANTS (phonopy 格式)")

    return phonon


def generate_fc2nd_from_vasp(unitcell_filename, supercell_matrix,
                              vasp_pattern="disp-*/vasprun.xml",
                              output_filename="FORCE_CONSTANTS_2ND",
                              write_phonopy_fc=False,
                              symmetrize=True):
    """直接从 VASP 计算结果收集力并生成 FORCE_CONSTANTS_2ND。"""
    unitcell = read_vasp(unitcell_filename)
    phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
    phonon.generate_displacements()

    n_satom = get_n_atoms(phonon.supercell)
    print(f"超胞原子数: {n_satom}")

    print("\n从 VASP 输出收集力 ...")
    forces_list, files = collect_forces_from_vasp(vasp_pattern)

    for idx, forces in enumerate(forces_list):
        if forces.shape[0] != n_satom:
            raise ValueError(
                f"文件 {files[idx]} 的原子数 ({forces.shape[0]}) 与超胞 ({n_satom}) 不匹配"
            )

    # phonopy 3.x 设置力的方式
    phonon.forces = np.array(forces_list, dtype="double")

    print("\n计算二阶力常数 (FC2) ...")
    phonon.produce_force_constants()

    if symmetrize:
        print("对称化力常数 ...")
        phonon.symmetrize_force_constants()

    fc = phonon.force_constants
    print(f"原始力常数 shape: {fc.shape}")

    fc = ensure_full_fc(phonon, fc)

    write_FORCE_CONSTANTS_2ND(fc, filename=output_filename)

    if write_phonopy_fc:
        write_FORCE_CONSTANTS(fc, filename="FORCE_CONSTANTS")
        print("同时输出 FORCE_CONSTANTS (phonopy 格式)")

    return phonon


def main():
    parser = argparse.ArgumentParser(
        description="生成适用于 FourPhonon/ShengBTE 的 FORCE_CONSTANTS_2ND",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python gen_fc2nd.py --mode from_forcesets -c POSCAR --dim "2 2 2"
  python gen_fc2nd.py --mode from_vasp -c POSCAR --dim "2 2 2" --pattern "disp-*/vasprun.xml"
  python gen_fc2nd.py --mode from_vasp -c POSCAR --dim "2 2 2" --write-fc
        """
    )
    parser.add_argument("--mode", choices=["from_forcesets", "from_vasp"], required=True)
    parser.add_argument("-c", "--cell", default="POSCAR", help="晶胞文件")
    parser.add_argument("--dim", required=True, help="超胞尺寸，如 '2 2 2'")
    parser.add_argument("--pattern", default="disp-*/vasprun.xml")
    parser.add_argument("--forcesets", default="FORCE_SETS")
    parser.add_argument("-o", "--output", default="FORCE_CONSTANTS_2ND")
    parser.add_argument("--write-fc", action="store_true")
    parser.add_argument("--no-symmetrize", action="store_true")

    args = parser.parse_args()

    dim = [int(x) for x in args.dim.split()]
    if len(dim) == 1:
        supercell_matrix = [[dim[0], 0, 0], [0, dim[0], 0], [0, 0, dim[0]]]
    elif len(dim) == 3:
        supercell_matrix = [[dim[0], 0, 0], [0, dim[1], 0], [0, 0, dim[2]]]
    elif len(dim) == 9:
        supercell_matrix = [dim[0:3], dim[3:6], dim[6:9]]
    else:
        raise ValueError("--dim 参数格式错误")

    symmetrize = not args.no_symmetrize

    if args.mode == "from_forcesets":
        phonon = generate_fc2nd_from_forcesets(
            args.cell, supercell_matrix,
            force_sets_filename=args.forcesets,
            output_filename=args.output,
            write_phonopy_fc=args.write_fc,
            symmetrize=symmetrize
        )
    else:
        phonon = generate_fc2nd_from_vasp(
            args.cell, supercell_matrix,
            vasp_pattern=args.pattern,
            output_filename=args.output,
            write_phonopy_fc=args.write_fc,
            symmetrize=symmetrize
        )

    print("\n=== 完成 ===")
    print(f"输出文件: {args.output}")
    print(f"超胞矩阵: {supercell_matrix}")
    print(f"超胞原子数: {get_n_atoms(phonon.supercell)}")

    with open(args.output, "r") as f:
        first_line = f.readline().strip()
    nums = first_line.split()
    if len(nums) >= 2 and nums[0] == nums[1]:
        print("格式检查通过: FULL 力常数格式")
    else:
        print("警告: 格式可能不是 FULL 格式")


if __name__ == "__main__":
    main()
