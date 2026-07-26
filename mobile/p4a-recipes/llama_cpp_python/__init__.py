"""python-for-android recipe for llama-cpp-python (on-device AI Coach).

llama-cpp-python wraps the llama.cpp C++ library and builds it via scikit-build
(CMake). The sdist filename on PyPI uses the NORMALISED package name
(``llama_cpp_python`` with underscores) -- the legacy ``/source/l/<letter>/``
path 302-redirects the underscore form to the real hash URL, but the hyphen
form 404s. A hyphen URL here was the real cause of the first CI build failure:
p4a retried the 404 ~5x then aborted ~40s in, never reaching CMake, and the CI
fallback grep (which matched bare "llama" in normal recipe-build-order output)
false-positive-stripped the recipe. See WARP.md "On-device AI Coach" notes.

The production path is a prebuilt Android wheel hosted in the ``yidaki53/
p4a-wheels`` index and pulled via ``--extra-index-url`` (p4a PR #3280, wired
through ``p4a.extra_args`` since buildozer 1.5.0 predates the first-class spec
tokens). This source recipe is the in-recipe fallback: ``PyProjectRecipe``
checks the index first and only falls back to this source build when no
prebuilt wheel matches. If the source build still fails, CI's build-apk job
strips this recipe and rebuilds without it; the AI Coach UI then degrades
gracefully (``is_llm_available()`` -> False -> informative message, no crash).

The build is CPU-only (no CUDA/Metal/Vulkan/OpenCL), matching the desktop
default. The model itself is NOT bundled -- it is downloaded on first use only
when the user opts in, keeping the APK small.
"""

from os.path import join

from pythonforandroid.recipe import PyProjectRecipe
from pythonforandroid.logger import info


class LlamaCppPythonRecipe(PyProjectRecipe):
    """Cross-compile llama-cpp-python (scikit-build + CMake) for Android.

    Used only when no prebuilt ``android_26_*`` wheel is found in the
    ``yidaki53/p4a-wheels`` index. Drives ``python -m build`` (scikit-build-core
    -> CMake) and injects the NDK CMake toolchain + CPU-only ggml flags via
    ``CMAKE_ARGS`` (the env var llama-cpp-python's CMake reads).
    """

    version = "0.3.14"
    # PyPI sdist: the filename uses the NORMALISED name (underscores). The
    # legacy ``/source/l/llama-cpp-python/`` path 302-redirects this underscore
    # form to the real hash URL; the hyphen form 404s (do NOT use hyphens here).
    url = "https://files.pythonhosted.org/packages/source/l/llama-cpp-python/llama_cpp_python-{version}.tar.gz"
    site_packages_name = "llama_cpp"
    depends = ["python3", "certifi"]
    # Runtime deps declared in llama-cpp-python's pyproject. numpy is already a
    # top-level requirement; these three are pure-Python and p4a installs them
    # via pip into the APK so ``Llama`` import/use works on-device.
    python_depends = ["typing-extensions", "diskcache", "jinja2"]
    # llama.cpp is C++; bundle libc++_shared so the .so files load on Android.
    need_stl_shared = True
    call_hostpython_via_targetpython = False

    # CPU-only ggml flags; disable every accelerator backend (none build cleanly
    # under the NDK without extra sysroots) and the features that need host tools.
    _GGML_CMAKE_ARGS = [
        "-DGGML_NATIVE=OFF",
        "-DGGML_OPENMP=OFF",
        "-DGGML_BLAS=OFF",
        "-DGGML_CUDA=OFF",
        "-DGGML_METAL=OFF",
        "-DGGML_VULKAN=OFF",
        "-DGGML_OPENCL=OFF",
        "-DGGML_SYCL=OFF",
        "-DGGML_RPC=OFF",
        "-DLLAMA_BUILD=ON",
        "-DLLAVA_BUILD=OFF",
        "-DBUILD_SHARED_LIBS=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    def get_recipe_env(self, arch, **kwargs):
        env = super().get_recipe_env(arch, **kwargs)
        ndk_dir = self.ctx.ndk_dir
        toolchain = join(ndk_dir, "build", "cmake", "android.toolchain.cmake")
        # llama-cpp-python's CMake reads the CMAKE_ARGS env var (see its
        # CMakeLists.txt / Makefile). Append the NDK toolchain + Android target
        # + CPU-only ggml flags so scikit-build passes them to CMake.
        cmake_args = list(self._GGML_CMAKE_ARGS) + [
            "-DCMAKE_TOOLCHAIN_FILE={}".format(toolchain),
            "-DANDROID_ABI={}".format(arch.arch),
            "-DANDROID_PLATFORM=android-{}".format(self.ctx.ndk_api),
            "-DANDROID_NDK={}".format(ndk_dir),
        ]
        existing = env.get("CMAKE_ARGS", "")
        env["CMAKE_ARGS"] = (
            (existing + ";" if existing else "") + ";".join(cmake_args)
        )
        # scikit-build/CMake also look for these directly.
        env["ANDROID_NDK_HOME"] = ndk_dir
        env["ANDROID_NDK"] = ndk_dir
        env["ANDROID_NDK_ROOT"] = ndk_dir
        env["ANDROID_ABI"] = arch.arch
        env["ANDROID_PLATFORM"] = "android-{}".format(self.ctx.ndk_api)
        return env

    def should_build(self, arch):
        # Avoid a redundant rebuild if the wheel is already installed.
        if self.ctx.has_package("llama_cpp"):
            info("llama_cpp already installed; skipping build")
            return False
        return True


recipe = LlamaCppPythonRecipe()
