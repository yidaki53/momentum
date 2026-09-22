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
                # CMake HARD-CODES /usr/include as an implicit include dir and
                # refuses to emit -I for it (cmLocalGenerator.cxx:
                # "implicitExclude.emplace(\"/usr/include\")"). So handing CMake
                # -DVulkan_INCLUDE_DIR=/usr/include is accepted by
                # find_package(Vulkan) but the resulting -I/usr/include is
                # silently dropped -- and the aarch64 build, which uses the NDK
                # --sysroot instead of the host /usr/include, still dies with
                # "'vulkan/vulkan.hpp' file not found" at ggml-vulkan.cpp:8.
                #
                # Fix: stage the distro Vulkan headers into a directory that is
                # NOT /usr/include, then point Vulkan_INCLUDE_DIR at that stage
                # so CMake emits -isystem <stage> for the ggml-vulkan target.
                #
                # Stage the ENTIRE Vulkan-Headers set (.h AND .hpp), not just the
                # .hpp files. vulkan.hpp carries
                #     static_assert( VK_HEADER_VERSION == <N>, ... )
                # for its OWN header version. If its <vulkan/vulkan.h> falls
                # through to the NDK sysroot it gets a DIFFERENT, older version
                # (NDK r25b ships 203 while the runner's distro Vulkan-Hpp is
                # 275), so the static_assert fires and every type added after 203
                # (VkVideoProfileInfoKHR, ...) is unknown -- that was the run-80
                # failure. Copying the whole set keeps vulkan.hpp and vulkan.h
                # version-consistent (both from the distro) and, since the distro
                # set is >= the sysroot's, a superset of what llama.cpp needs.
                stage_inc = join(self.get_build_dir(arch.arch),
                                 "vulkan-headers-stage")
                ensure_dir(stage_inc)
                # The Vulkan-Headers install lays out TWO sibling include dirs:
                #   <root>/vulkan/*.h,*.hpp      <root>/vk_video/*.h
                # vulkan_core.h pulls the video codec headers from the sibling
                # vk_video/ dir (e.g.
                # #include "vk_video/vulkan_video_codec_h264std.h"), so BOTH
                # must be staged or the target build dies with
                # "'vk_video/vulkan_video_codec_h264std.h' file not found".
                for _sub in ("vulkan", "vk_video"):
                    _src = join(distro_vulkan_inc, _sub)
                    if not os.path.isdir(_src):
                        continue
                    _dst = join(stage_inc, _sub)
                    ensure_dir(_dst)
                    for _hdr in glob.glob(join(_src, "*")):
                        if isfile(_hdr):
                            shutil.copy2(_hdr, join(_dst, basename(_hdr)))
                cmake_args.append("-DVulkan_INCLUDE_DIR={}".format(stage_inc))
                cmake_args.append("-DCMAKE_INCLUDE_PATH={}".format(stage_inc))
                info("Vulkan C++ headers staged {} -> {}".format(
                    distro_vulkan_inc, stage_inc))
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
        # the same invocation. CRITICAL: the config-setting carries ALL flags that
        # the nested CMake configures need, including the vulkan-specific ones
        # (-DGGML_VULKAN=ON, -DVulkan_INCLUDE_DIR, -DCMAKE_INCLUDE_PATH) which are
        # NOT reliably delivered via CMAKE_ARGS env var (scikit-build-core emits
        # "Unsupported CMAKE_ARGS ignored" for each -D flag in that env var).
        #
        # Build the cmake.args string from the same cmake_args list used for
        # CMAKE_ARGS, plus the static-library try-compile safeguard (which must
        # reach the nested host-toolchain configure too). Note: CMake_TRY_COMPILE
        # flags are CMAKE_ARGS-style and won't be read from cmake.args by the
        # host tool config, but we pass them anyway as belt-and-suspenders; the
        # LDFLAGS strip (above) is the real mechanism that fixes the host tool
        # link failure.
        self.extra_build_args = [
            "--config-setting",
            "cmake.args=" + ";".join(
                existing_flags + cmake_args + [
                    "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
                ]
            ),
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
