"""python-for-android recipe for llama-cpp-python (on-device AI Coach).

HIGH RISK / BEST-EFFORT: llama-cpp-python wraps the llama.cpp C++ library and
builds it via scikit-build (CMake). There is no official Android wheel, so this
recipe cross-compiles llama.cpp for Android via the NDK CMake toolchain. The
scikit-build/CMake env propagation is the fragile point (the same class of
failure that killed the pydantic-core/maturin approach), so the CI build-apk
job includes a fallback that strips this recipe and rebuilds without it; the AI
Coach UI degrades gracefully when the native backend is absent
(``is_llm_available()`` -> False -> informative message, no crash). See
WARP.md "AI Coach on Android" notes and ``.github/workflows/ci.yml``.

The build is CPU-only (no CUDA/Metal/Vulkan/OpenCL), matching the desktop
default. The model itself is NOT bundled -- it is downloaded on first use only
when the user opts in, keeping the APK small.
"""

from os.path import join

from pythonforandroid.recipe import PyProjectRecipe
from pythonforandroid.logger import info


class LlamaCppPythonRecipe(PyProjectRecipe):
    """Cross-compile llama-cpp-python (scikit-build + CMake) for Android.

    Uses ``PyProjectRecipe`` to drive ``python -m build`` (which invokes
    scikit-build-core -> CMake), and injects the NDK CMake toolchain + CPU-only
    ggml flags via ``CMAKE_ARGS`` (the env var llama-cpp-python's CMake reads).
    """

    version = "0.3.14"
    # PyPI source distribution (includes the vendored llama.cpp source tree).
    url = "https://files.pythonhosted.org/packages/source/l/llama-cpp-python/llama-cpp-python-{version}.tar.gz"
    site_packages_name = "llama_cpp"
    depends = ["python3", "certifi"]
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
