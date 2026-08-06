#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Phonopy 的 FORCE_SETS 或 VASP 计算结果生成适用于 FourPhonon/ShengBTE 的 FORCE_CONSTANTS_2ND。

兼容 phonopy >= 1.13.2，利用 phonopy Python API 直接处理。

使用方法:
 方式1: 从已有 FORCE_SETS 生成 FORCE_CONSTANTS_2ND
 python gen_fc2nd.py --mode from_forcesets --dim "2 2 2" -c POSCAR

 方式2: 直接从 disp-*/vasprun.xml 收集力并生成 FORCE_CONSTANTS_2ND
 python gen_fc2nd.py --mode from_vasp --dim "2 2 2" -c POSCAR --pattern "disp-*/vasprun.xml"

输出:
 FORCE_CONSTANTS_2ND (FourPhonon/ShengBTE 兼容格式)
 FORCE_CONSTANTS (phonopy 原生格式，可选)

修复记录 (2026-08-06):
 v1 错误: 手动 compact->full 扩展，平移对称性映射错误
 v2 问题: 默认执行 symmetrize_force_constants()，与 phonopy 命令行
          --writefc --full-fc 默认行为不一致（命令行默认不对称化）
 v3 修复: 默认关闭对称化，添加 --symmetrize 选项。默认输出与 phonopy
          命令行完全一致。用户需要对称化时可显式开启。
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
    print(f"错误: 无法导入 phonopy 模块。请确保已安装 phonopy >= 1.13.2。\n{e}")
    sys.exit(1)


def get_n_atoms(cell_obj):
    """兼容不同 phonopy 版本获取原子数。"""
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
    格式: 每对 (i,j) 占 4 行，第一行为原子编号，后三行为 3x3 张量。
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


def generate_fc2nd_from_forcesets(unitcell_filename, supercell_matrix,
                                   force_sets_filename="FORCE_SETS",
                                   output_filename="FORCE_CONSTANTS_2ND",
                                   write_phonopy_fc=False,
                                   symmetrize=False):
    """从已有的 FORCE_SETS 生成 FORCE_CONSTANTS_2ND。"""
    unitcell = read_vasp(unitcell_filename)
    phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
    phonon.generate_displacements()

    n_satom = get_n_atoms(phonon.supercell)
    print(f"超胞原子数: {n_satom}")

    print(f"读取 {force_sets_filename} ...")
    try:
        dataset = parse_FORCE_SETS(natom=n_satom, filename=force_sets_filename)
    except TypeError:
        dataset = parse_FORCE_SETS(filename=force_sets_filename)
    phonon.dataset = dataset

    print("计算二阶力常数 (FC2) ...")
    # 显式要求 phonopy 计算 full 格式力常数
    phonon.produce_force_constants(calculate_full_force_constants=True)

    if symmetrize:
        print("对称化力常数 (permutation symmetry) ...")
        phonon.symmetrize_force_constants()
    else:
        print("跳过对称化 (与 phonopy 命令行默认行为一致)")

    fc = phonon.force_constants
    print(f"力常数 shape: {fc.shape}")

    # 验证确实是 full 格式
    if fc.shape[0] != n_satom or fc.shape[1] != n_satom:
        raise RuntimeError(
            f"力常数不是 full 格式: 期望 ({n_satom},{n_satom},3,3), 实际 {fc.shape}"
        )

    write_FORCE_CONSTANTS_2ND(fc, filename=output_filename)

    if write_phonopy_fc:
        write_FORCE_CONSTANTS(fc, filename="FORCE_CONSTANTS")
        print("同时输出 FORCE_CONSTANTS (phonopy 格式)")

    return phonon


def generate_fc2nd_from_vasp(unitcell_filename, supercell_matrix,
                              vasp_pattern="disp-*/vasprun.xml",
                              output_filename="FORCE_CONSTANTS_2ND",
                              write_phonopy_fc=False,
                              symmetrize=False):
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

    phonon.forces = np.array(forces_list, dtype="double")

    print("\n计算二阶力常数 (FC2) ...")
    phonon.produce_force_constants(calculate_full_force_constants=True)

    if symmetrize:
        print("对称化力常数 (permutation symmetry) ...")
        phonon.symmetrize_force_constants()
    else:
        print("跳过对称化 (与 phonopy 命令行默认行为一致)")

    fc = phonon.force_constants
    print(f"力常数 shape: {fc.shape}")

    # 验证确实是 full 格式
    if fc.shape[0] != n_satom or fc.shape[1] != n_satom:
        raise RuntimeError(
            f"力常数不是 full 格式: 期望 ({n_satom},{n_satom},3,3), 实际 {fc.shape}"
        )

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
  # 默认行为: 与 phonopy --writefc --full-fc 一致（不对称化）
  python gen_fc2nd.py --mode from_forcesets -c POSCAR --dim "2 2 2"

  # 需要对称化时显式开启
  python gen_fc2nd.py --mode from_forcesets -c POSCAR --dim "2 2 2" --symmetrize

  # 从 VASP 输出直接生成
  python gen_fc2nd.py --mode from_vasp -c POSCAR --dim "2 2 2" --pattern "disp-*/vasprun.xml"
        """
    )
    parser.add_argument("--mode", choices=["from_forcesets", "from_vasp"], required=True)
    parser.add_argument("-c", "--cell", default="POSCAR", help="晶胞文件")
    parser.add_argument("--dim", required=True, help="超胞尺寸，如 '2 2 2'")
    parser.add_argument("--pattern", default="disp-*/vasprun.xml")
    parser.add_argument("--forcesets", default="FORCE_SETS")
    parser.add_argument("-o", "--output", default="FORCE_CONSTANTS_2ND")
    parser.add_argument("--write-fc", action="store_true")
    # v3 关键修改: 默认关闭对称化，改为 --symmetrize 选项
    parser.add_argument("--symmetrize", action="store_true",
                        help="对力常数施加置换对称性 (默认关闭，与phonopy命令行一致)")

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

    if args.mode == "from_forcesets":
        phonon = generate_fc2nd_from_forcesets(
            args.cell, supercell_matrix,
            force_sets_filename=args.forcesets,
            output_filename=args.output,
            write_phonopy_fc=args.write_fc,
            symmetrize=args.symmetrize
        )
    else:
        phonon = generate_fc2nd_from_vasp(
            args.cell, supercell_matrix,
            vasp_pattern=args.pattern,
            output_filename=args.output,
            write_phonopy_fc=args.write_fc,
            symmetrize=args.symmetrize
        )

    print("\n=== 完成 ===")
    print(f"输出文件: {args.output}")
    print(f"超胞矩阵: {supercell_matrix}")
    print(f"超胞原子数: {get_n_atoms(phonon.supercell)}")
    print(f"对称化: {'已启用' if args.symmetrize else '未启用 (默认)'}")

    with open(args.output, "r") as f:
        first_line = f.readline().strip()
        nums = first_line.split()
        if len(nums) >= 2 and nums[0] == nums[1]:
            print("格式检查通过: FULL 力常数格式")
        else:
            print("警告: 格式可能不是 FULL 格式")


if __name__ == "__main__":
    main()
