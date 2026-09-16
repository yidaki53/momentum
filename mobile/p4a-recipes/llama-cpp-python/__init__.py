"""python-for-android recipe for llama-cpp-python (on-device AI Coach).

llama-cpp-python wraps the llama.cpp C++ library and builds it via scikit-build
(CMake). The sdist filename on PyPI uses the NORMALISED package name
(llama_cpp_python with underscores) -- the legacy /source/l/<letter>/ path
302-redirects the underscore form to the real hash URL, but the hyphen form
404s. See WARP.md "On-device AI Coach" notes.

The production path is a prebuilt Android wheel hosted in the yidaki53/p4a-wheels
index and pulled via --extra-index-url (p4a PR #3280, wired through p4a.extra_args
since buildozer 1.5.0 predates the first-class spec tokens). This source recipe
is the in-recipe fallback: PyProjectRecipe checks the index first and only falls
back to this source build when no prebuilt wheel matches. The build must succeed
with llama-cpp-python included -- no fallback to a build without on-device inference.

The build is CPU-only (no CUDA/Metal/Vulkan/OpenCL), matching the desktop
default. The model itself is NOT bundled -- it is downloaded on first use only
when the user opts in, keeping the APK small.

Build-system notes
------------------
p4a clones python-for-android fresh from GitHub and its PyProjectRecipe.build_arch
invokes `python -m build --wheel --config-setting builddir=<dir>`. That config-setting
key (builddir, no hyphen) is what *setuptools-backed* backends expect, but
llama-cpp-python 0.3.14 builds with scikit-build-core 1.x, whose recognised key is
`build-dir` (hyphenated). Passing `builddir` makes scikit-build-core abort::

    ERROR: Unrecognized options in config-settings:
      builddir -> Did you mean: build-dir, build?

scikit-build-core emits that error before the source build starts, so the fix must
replace the bad key rather than append around it. Keep this recipe as close to
upstream `PyProjectRecipe.build_arch` as possible: the only behavioural changes
are the corrected scikit-build-core `build-dir` config-setting and the NDK/CMake
environment from `get_recipe_env`.
"""

import glob
import sh
from os.path import join, isfile, realpath, basename

from pythonforandroid.util import current_directory, ensure_dir
from pythonforandroid.recipe import PyProjectRecipe
from pythonforandroid.logger import warning, shprint, info


class LlamaCppPythonRecipe(PyProjectRecipe):
    """Cross-compile llama-cpp-python (scikit-build + CMake) for Android.

    Used only when no prebuilt android_26_* wheel is found in the
    yidaki53/p4a-wheels index. Drives `python -m build` (scikit-build-core ->
    CMake) and injects the NDK CMake toolchain + CPU-only ggml flags via
    CMAKE_ARGS (the env var llama-cpp-python's CMake reads).
    """

    version = "0.3.14"
    url = "https://files.pythonhosted.org/packages/source/l/llama-cpp-python/llama_cpp_python-{version}.tar.gz"
    site_packages_name = "llama_cpp"
    depends = ["python3", "certifi"]
    python_depends = ["typing-extensions", "diskcache", "jinja2"]
    need_stl_shared = True
    call_hostpython_via_targetpython = False

    _GGML_CMAKE_ARGS = [
        "-DGGML_NATIVE=OFF", "-DGGML_OPENMP=OFF", "-DGGML_BLAS=OFF",
        "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF",
        "-DGGML_OPENCL=OFF", "-DGGML_SYCL=OFF", "-DGGML_RPC=OFF",
        "-DLLAMA_BUILD=ON", "-DLLAVA_BUILD=OFF", "-DBUILD_SHARED_LIBS=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    def get_recipe_env(self, arch, **kwargs):
        env = super().get_recipe_env(arch, **kwargs)
        ndk_dir = self.ctx.ndk_dir
        toolchain = join(ndk_dir, "build", "cmake", "android.toolchain.cmake")
        cmake_args = list(self._GGML_CMAKE_ARGS) + [
            "-DCMAKE_TOOLCHAIN_FILE={}".format(toolchain),
            "-DANDROID_ABI={}".format(arch.arch),
            "-DANDROID_PLATFORM=android-{}".format(self.ctx.ndk_api),
            "-DANDROID_NDK={}".format(ndk_dir),
        ]
        existing = env.get("CMAKE_ARGS", "")
        if existing:
            # bootstrap cmake args are space-separated -D flags; merge into a
            # single space-separated CMAKE_ARGS string so CMake/CMakeToolchain
            # parses each -D flag individually (not as a single joined value).
            existing_flags = existing.split()
        else:
            existing_flags = []
        env["CMAKE_ARGS"] = " ".join(existing_flags + cmake_args)
        env["ANDROID_NDK_HOME"] = ndk_dir
        env["ANDROID_NDK"] = ndk_dir
        env["ANDROID_NDK_ROOT"] = ndk_dir
        env["ANDROID_ABI"] = arch.arch
        env["ANDROID_PLATFORM"] = "android-{}".format(self.ctx.ndk_api)
        return env

    def build_arch(self, arch):
        # Mirror PyProjectRecipe.build_arch, except that the `python -m build`
        # config-setting key is scikit-build-core's `build-dir` (hyphenated) rather
        # than upstream's `builddir`. scikit-build-core rejects the un-hyphenated
        # key outright ("Unrecognized options in config-settings"), so the bad
        # key must be replaced, not appended to. All other behaviour -- prebuilt
        # short-circuit, hostpython prerequisites, env from get_recipe_env, and
        # install_wheel -- is unchanged from upstream.
        if self.check_prebuilt(arch, "skipping build_arch"):
            result = self.install_prebuilt_wheel(arch)
            if result:
                return
            warning("Failed to install prebuilt wheel, falling back to build_arch")

        build_dir = self.get_build_dir(arch.arch)
        if not (isfile(join(build_dir, "pyproject.toml")) or isfile(join(build_dir, "setup.py"))):
            warning("Skipping build because it does not appear to be a Python project.")
            return
        self.install_hostpython_prerequisites(
            packages=["build[virtualenv]", "pip", "setuptools", "patchelf"] + self.hostpython_prerequisites
        )

        env = self.get_recipe_env(arch, with_flags_in_cc=True)
        sub_build_dir = join(build_dir, "p4a_android_build")
        ensure_dir(sub_build_dir)

        # scikit-build-core accepts `build-dir` (hyphenated), not `builddir`.
        build_args = [
            "-m", "build", "--wheel",
            "--config-setting", "build-dir={}".format(sub_build_dir),
        ] + self.extra_build_args

        built_wheels = []
        with current_directory(build_dir):
            shprint(
                sh.Command(self.real_hostpython_location), *build_args, _env=env
            )
            built_wheels = [realpath(whl) for whl in glob.glob("dist/*.whl")]
        self.install_wheel(arch, built_wheels)

    def install_libraries(self, arch):
        """Stage llama_cpp's native backends onto the Android loader path.

        p4a's ``PyProjectRecipe.install_wheel()`` only unpacks the wheel into
        the python install dir, which gets bundled into ``libpybundle.so``.
        But ``llama_cpp`` loads its backends with ``ctypes`` via
        ``os.path.dirname(__file__) / "lib" / lib<ggml>.so`` and ``libllama.so``
        has NO rpath while its ``DT_NEEDED`` list imports ``libggml.so``,
        ``libggml-base.so`` and ``libggml-cpu.so``. Android's linker only
        resolves sibling libraries that live in the app's native lib dir
        (``lib/<arch>/``), NOT the package dir, so the libs must ALSO be
        staged there.

        ``install_libraries()`` is called by p4a *unconditionally* after
        ``build_arch`` (even when build_arch was skipped on a cache hit), so
        overriding it here guarantees the libs land in ``libs/<arch>/`` that
        gradle ships into the APK's ``lib/<arch>/`` -- i.e. on the loader path.
        Without this, ``from llama_cpp import Llama`` raises at runtime and the
        AI Coach disables itself ("inference is not available on this build").
        """
        # Defer to base first: handles built_libraries recipes (no-op here).
        super().install_libraries(arch)
        destination = self.ctx.get_python_install_dir(arch.arch)
        native_libs = glob.glob(join(destination, "llama_cpp", "lib", "*.so"))
        if native_libs:
            info("Staging llama native libs into native lib dir: "
                 + ", ".join(sorted(basename(p) for p in native_libs)))
            self.install_libs(arch, *native_libs)


recipe = LlamaCppPythonRecipe()
