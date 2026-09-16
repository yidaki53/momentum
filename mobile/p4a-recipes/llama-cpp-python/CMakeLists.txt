# GPU backend support (opt-in; CUDA is never enabled on Android/aarch64 builds).
# CUDA is always OFF to keep builds deterministic and device-local.
set(GGML_CUDA OFF)
# Vulkan is buildable (NDK r25b ships vulkan/vulkan.h + libvulkan.so for
# aarch64), but disabled by default for size/stability. Enable by passing
# -DGGML_VULKAN=ON on the cmake invocation (or export GGML_VULKAN=ON).
if(DEFINED GGML_VULKAN)
    set(GGML_VULKAN ${GGML_VULKAN})
else()
    set(GGML_VULKAN OFF)
endif()