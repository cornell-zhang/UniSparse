# Autogenerated from /home/dingzj/sparse/Sparse_Layout_Dialect/test/lit.site.cfg.py.in
# Do not edit!

# Allow generated file to be relocatable.
from pathlib import Path
def path(p):
    if not p: return ''
    return str((Path(__file__).parent / p).resolve())


import sys

config.host_triple = "x86_64-unknown-linux-gnu"
config.target_triple = "x86_64-unknown-linux-gnu"
config.llvm_src_root = ""
config.llvm_obj_root = "/home/dingzj/sparse/tmp/LLVM/build"
config.llvm_tools_dir = "/home/dingzj/sparse/tmp/LLVM/build/./bin"
config.llvm_lib_dir = "/home/dingzj/sparse/tmp/LLVM/build/./lib"
config.llvm_shlib_dir = ""
config.llvm_shlib_ext = ".so"
config.llvm_exe_ext = ""
config.lit_tools_dir = ""
config.python_executable = ""
config.gold_executable = ""
config.ld64_executable = ""
config.enable_shared = 1
config.enable_assertions = 1
config.targets_to_build = " X86 NVPTX AMDGPU"
config.native_target = "X86"
config.llvm_bindings = "".split(' ')
config.host_os = "Linux"
config.host_cc = "/usr/bin/clang "
config.host_cxx = "/usr/bin/clang++ "
config.enable_libcxx = ""
# Note: ldflags can contain double-quoted paths, so must use single quotes here.
config.host_ldflags = ''
config.llvm_use_sanitizer = ""
config.llvm_host_triple = 'x86_64-unknown-linux-gnu'
config.host_arch = "x86_64"
config.sparlay_src_root = "/home/dingzj/sparse/Sparse_Layout_Dialect"
config.sparlay_obj_root = "/home/dingzj/sparse/Sparse_Layout_Dialect/build"

# Support substitution of the tools_dir with user parameters. This is
# used when we can't determine the tool dir at configuration time.
try:
    config.llvm_tools_dir = config.llvm_tools_dir % lit_config.params
    config.llvm_lib_dir = config.llvm_lib_dir % lit_config.params
    config.llvm_shlib_dir = config.llvm_shlib_dir % lit_config.params
except KeyError:
    e = sys.exc_info()[1]
    key, = e.args
    lit_config.fatal("unable to find %r parameter, use '--param=%s=VALUE'" % (key,key))


import lit.llvm
lit.llvm.initialize(lit_config, config)

# Let the main config do the real work.
lit_config.load_config(config, "/home/dingzj/sparse/Sparse_Layout_Dialect/test/lit.cfg.py")