from setuptools import setup
from Cython.Build import cythonize
import sys

# 设置 MSVC 编译选项
if sys.platform == "win32":
    extra_compile_args = ['/O2', '/Wall', '/GL']  # 启用最大优化和链接时优化
else:
    extra_compile_args = ['-O3', '-march=native', '-flto']  # 对于 Linux/macOS 使用 LTO

setup(
    ext_modules=cythonize(
        "tools.pyx",  # Cython 文件路径
        compiler_directives={
            'language_level': 3,   # 使用 Python 3 语法
        },
        annotate=False,  # 禁用生成 .html 注释文件
        compile_time_env={'CFLAGS': extra_compile_args},
        nthreads=10  # 使用 4 个线程来并行编译
    )
)
