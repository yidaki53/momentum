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

The default build is CPU-only (no CUDA/Metal/Vulkan/OpenCL), matching the
desktop default. Vulkan (GPU) inference is opt-in: set the env var
``GGML_VULKAN=ON`` to enable the Vulkan backend, which forces a source build
because the prebuilt wheels in the yidaki53/p4a-wheels index are CPU-only.
The model itself is NOT bundled -- it is downloaded on first use only
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
import os
import sh
import shutil
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
        "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF",
        "-DGGML_OPENCL=OFF", "-DGGML_SYCL=OFF", "-DGGML_RPC=OFF",
        "-DLLAMA_BUILD=ON", "-DLLAVA_BUILD=OFF", "-DBUILD_SHARED_LIBS=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        # Cross-compile safeguard: scikit-build-core runs a secondary CMake
        # configure with the *host* compiler, and p4a injects the target
        # (aarch64) Python into the link flags (-L<python3 android-build> -lpython3.14).
        # The host C-compiler test then links the aarch64 libpython3.14.so with the
        # host linker and aborts with "incompatible with elf64-x86-64". Forcing
        # try_compiles to produce static archives (compile-only, no link) makes the
        # compiler test pass; the real aarch64 libs are still built with the NDK
        # toolchain (the active ninja invocations target aarch64-none-linux-android).
        "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
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
        # Vulkan (GPU) inference is opt-in. Default OFF keeps the CPU-only
        # production path; the dedicated CI job sets GGML_VULKAN=ON to build the
        # Vulkan APK artifact, and build_arch skips the prebuilt CPU wheel below.
        vulkan = os.environ.get("GGML_VULKAN", "OFF")
        if str(vulkan).upper() in ("ON", "1", "TRUE"):
            cmake_args.append("-DGGML_VULKAN=ON")
            # VULKAN_SDK must point at a directory whose parent contains a
            # usable glslc: llama.cpp's CMake looks for glslc under
            # ``$VULKAN_SDK/../bin`` and ``$VULKAN_SDK/bin``. The NDK's Vulkan
            # sources have no bin/ sibling, so prefer an explicit GLSLC
            # location: NDK shader-tools first, then distro glslc. Without a
            # compiler on PATH the Vulkan build fails deep in CMake instead of
            # here with a clear message.
            ndk_glslc_dir = join(ndk_dir, "shader-tools", "linux-x86_64")
            ndk_glslc = join(ndk_glslc_dir, "glslc")
            distro_glslc = shutil.which("glslc")
            if os.path.isfile(ndk_glslc) and os.access(ndk_glslc, os.X_OK):
                env["PATH"] = ndk_glslc_dir + os.pathsep + env.get("PATH", "")
                env["GLSLC"] = ndk_glslc
                info("Using NDK-bundled glslc: " + ndk_glslc)
            elif distro_glslc:
                env["GLSLC"] = distro_glslc
                info("Using distro glslc: " + distro_glslc)
            else:
                raise RuntimeError(
                    "GGML_VULKAN=ON requires glslc (NDK shader-tools or "
                    "glslang-tools package); refusing to start a doomed "
                    "Vulkan source build."
                )
                                    # The NDK's Vulkan sources (sources/third_party/vulkan) ship only the
            # C header ``vulkan/vulkan.h`` -- used by llama.cpp's
            # vulkan-shaders-gen HOST tool at configure, and correctly ABI-matched
            # for the cross-compile. The target build, however, compiles
            # ggml-vulkan.cpp which does ``#include <vulkan/vulkan.hpp>`` (the
            # C++ Vulkan-Hpp header from the vulkan-headers package). CMake's
            # find_package(Vulkan) resolves the loader lib (libvulkan.so from the
            # NDK sysroot) but does NOT add the vulkan.hpp-bearing include dir, so
            # the target build dies with "fatal error: 'vulkan/vulkan.hpp' file
            # not found" once it reaches ggml-vulkan.cpp (~[9/45]).
            #
            # Fix: keep VULKAN_SDK pointing at the NDK (for the loader lib) but
            # also hand llama.cpp's CMake the distro Vulkan-Headers include dir
            # via -DVulkan_INCLUDE_DIR, and add it to the cross-compile search
            # path so the C++ header is visible to the aarch64 target build.
            env["VULKAN_SDK"] = join(ndk_dir, "sources", "third_party", "vulkan")
            distro_vulkan_inc = None
            for _cand in ("/usr/include", "/usr/local/include"):
                if os.path.isfile(join(_cand, "vulkan", "vulkan.hpp")):
                    distro_vulkan_inc = _cand
                    break
            if distro_vulkan_inc:
                cmake_args.append(
                    "-DVulkan_INCLUDE_DIR={}".format(distro_vulkan_inc))
                cmake_args.append(
                    "-DCMAKE_INCLUDE_PATH={}".format(distro_vulkan_inc))
                info("Vulkan C++ headers (vulkan/vulkan.hpp): "
                     + distro_vulkan_inc)
            else:
                warning("vulkan/vulkan.hpp not found in /usr/include or "
                        "/usr/local/include; ggml-vulkan.cpp will fail to "
                        "compile. Install the vulkan-headers package.")

            # --- Vulkan host-toolchain link fix ---
            # scikit-build-core builds llama.cpp's vulkan-shaders-gen HOST tool
            # (a tiny C program) with the system gcc via a generated
            # host-toolchain.cmake, and llama.cpp drives it through an
            # ExternalProject_Add -- a *separate* CMake configure that the
            # `--config-setting cmake.args=...` above (which only reaches
            # scikit-build-core's main configure) cannot reach.
            #
            # p4a's LDFLAGS still carries the *target* (aarch64) Python link
            # flags from the python3 recipe (`-L.../python3/.../android-build`
            # `-lpython3.14`). CMake injects LDFLAGS into the host compiler-ABI
            # check's link line, so the host gcc tries to link the aarch64
            # libpython3.14.so and aborts "incompatible with elf64-x86-64"
            # (cmTC_0d23b -> "Check for working C compiler: ... - broken").
            #
            # llama.cpp's CMake outputs (libllama.so + the ggml backends) never
            # reference any Python symbol and so never need -lpython<ver> on the
            # link line; strip it so the host tool's compiler check links cleanly
            # with libc. CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY can't be
            # exported via env (CMake ignores it there), so the only lever that
            # reaches the nested host configure is LDFLAGS itself.
            ldflags = env.get("LDFLAGS", "").split()
            ldflags = [f for f in ldflags if not f.startswith("-lpython")]
            env["LDFLAGS"] = " ".join(ldflags)
        else:
            cmake_args.append("-DGGML_VULKAN=OFF")
        existing = env.get("CMAKE_ARGS", "")
        if existing:
            # bootstrap cmake args are space-separated -D flags; merge into a
            # single space-separated CMAKE_ARGS string so CMake/CMakeToolchain
            # parses each -D flag individually (not as a single joined value).
            existing_flags = existing.split()
        else:
            existing_flags = []
        env["CMAKE_ARGS"] = " ".join(existing_flags + cmake_args)

        # Also pass the CMake flags through `--config-setting` so scikit-build-core
        # (which p4a invokes via `python -m build` in build_arch) receives them via
        # the supported PEP 517 mechanism. llama-cpp-python reads CMAKE_ARGS (env)
        # AND config-settings; p4a sets CMAKE_ARGS in get_recipe_env, but the
        # scikit-build-core config-setting path is the most reliable carrier for
        # -D flags through `python -m build`. The extra_build_args list is appended
        # to the --config-setting build-dir=... args in build_arch, so these join
        # the same invocation. (GGML_VULKAN is already set in the cmake_args list
        # above; only the host try-compile safeguard and build type are repeated here.)
        self.extra_build_args = [
            "--config-setting", "cmake.args=-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY;-DCMAKE_BUILD_TYPE=Release",
        ]

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
        # Prebuilt wheels in the yidaki53/p4a-wheels index are CPU-only
        # (-DGGML_VULKAN=OFF); a Vulkan build must skip the prebuilt short-circuit
        # and compile llama-cpp-python from source so the Vulkan backend links in.
        if os.environ.get("GGML_VULKAN", "OFF").upper() in ("ON", "1", "TRUE"):
            info("Vulkan requested: ignoring prebuilt CPU wheel, building from source")
        elif self.check_prebuilt(arch, "skipping build_arch"):
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
