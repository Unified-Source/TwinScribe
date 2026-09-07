"""Tests for the machine probe and the acceleration plan: the pure policy on hand-built
inputs, the probes returning documented shapes on this platform, and the CUDA registration
being a no-op without the runtime packages."""

from __future__ import annotations

import pytest

from twinscribe import hardware
from twinscribe.hardware import (
    ARCH_ARM64,
    ARCH_OTHER,
    ARCH_X86_64,
    BACKEND_CT2,
    BACKEND_NONE,
    BACKEND_ONNX,
    DEVICE_CPU,
    DEVICE_CUDA,
    OS_LINUX,
    OS_MACOS,
    OS_WINDOWS,
    Gpu,
    Libraries,
    Machine,
    choose_compute_type,
    default_thread_count,
    describe_machine,
    make_plan,
    normalise_arch,
    os_key,
)


def machine(os_name: str = OS_WINDOWS, arch: str = ARCH_X86_64, cores: int = 16, memory: float | None = 32.0) -> Machine:
    return Machine(os=os_name, arch=arch, machine_raw=arch, cores=cores, memory_gb=memory, processor="test")


def libraries(ct2: bool = True, sherpa: bool = True, sherpa_cuda: bool = False) -> Libraries:
    versions = {}
    if sherpa:
        versions["sherpa-onnx"] = "1.13.7+cuda" if sherpa_cuda else "1.13.7"
    if ct2:
        versions["ctranslate2"] = "4.8.2"
        versions["faster-whisper"] = "1.2.1"
    return Libraries(faster_whisper=ct2, ctranslate2=ct2, sherpa_onnx=sherpa, onnxruntime=sherpa, versions=versions)


def test_arch_and_os_normalisation() -> None:
    for spelling in ("AMD64", "x86_64", "x64", "amd64"):
        assert normalise_arch(spelling) == ARCH_X86_64
    for spelling in ("ARM64", "aarch64", "arm64", "armv8l"):
        assert normalise_arch(spelling) == ARCH_ARM64
    assert normalise_arch("riscv64") == ARCH_OTHER and normalise_arch(None) == ARCH_OTHER
    assert os_key("win32") == OS_WINDOWS and os_key("linux") == OS_LINUX and os_key("darwin") == OS_MACOS
    assert os_key("freebsd13") == "freebsd13"


def test_thread_default_and_compute_type() -> None:
    assert default_thread_count(4) == 4 and default_thread_count(32) == 8 and default_thread_count(32, 12) == 12
    with pytest.raises(ValueError):
        default_thread_count(4, 0)
    assert choose_compute_type(DEVICE_CPU, {"int8", "float32"}) == "int8"
    assert choose_compute_type(DEVICE_CPU, {"float32"}) == "float32"
    assert choose_compute_type(DEVICE_CUDA, {"int8_float16", "float32"}) == "int8_float16"
    assert choose_compute_type(DEVICE_CUDA, set()) == "auto"


def test_plan_processor_only_x86() -> None:
    plan = make_plan(machine(), libraries(), ct2_types_cpu={"int8", "float32"})
    assert plan.detector_backend == BACKEND_CT2
    assert plan.detector is not None and plan.detector.device == DEVICE_CPU and plan.detector.compute_type == "int8"
    assert plan.publisher.provider == "cpu" and plan.diarizer.provider == "cpu"
    assert plan.parallel is False and plan.detector.threads == 8
    assert plan.platform_key == "windows-x86_64"
    assert "Detector (Whisper): CTranslate2 on processor, int8, 8 threads" in plan.describe()[0]


def test_plan_uses_cuda_for_the_detector_and_runs_engines_in_parallel() -> None:
    gpu = (Gpu(name="Test GPU", memory_mb=8192, driver="1.0"),)
    plan = make_plan(machine(), libraries(), gpus=gpu, cuda_count=1, ct2_types_cuda={"float16", "int8_float16"})
    assert plan.detector is not None and plan.detector.device == DEVICE_CUDA and plan.detector.compute_type == "float16"
    assert plan.publisher.device == DEVICE_CPU and plan.parallel is True
    assert any("processor build" in note for note in plan.notes)
    both = make_plan(machine(), libraries(sherpa_cuda=True), gpus=gpu, cuda_count=1, ct2_types_cuda={"float16"})
    assert both.publisher.provider == "cuda" and both.diarizer.device == DEVICE_CUDA and both.parallel is False


def test_plan_respects_the_processor_preference_and_reports_an_unusable_gpu() -> None:
    gpu = (Gpu(name="Test GPU", memory_mb=None, driver=None),)
    cpu_only = make_plan(machine(), libraries(sherpa_cuda=True), gpus=gpu, cuda_count=1, preference=DEVICE_CPU, ct2_types_cpu={"int8"})
    assert cpu_only.detector is not None and cpu_only.detector.device == DEVICE_CPU
    assert cpu_only.publisher.provider == "cpu" and cpu_only.parallel is False
    unusable = make_plan(machine(), libraries(), gpus=gpu, cuda_count=0, ct2_types_cpu={"int8"})
    assert unusable.detector is not None and unusable.detector.device == DEVICE_CPU
    assert any("CTranslate2 cannot use it" in note for note in unusable.notes)
    asked = make_plan(machine(), libraries(), preference=DEVICE_CUDA, cuda_count=0)
    assert any("asked for but none is usable" in note for note in asked.notes)
    with pytest.raises(ValueError):
        make_plan(machine(), libraries(), preference="tpu")


def test_plan_falls_back_to_onnx_on_windows_arm_and_to_none_without_libraries() -> None:
    arm = make_plan(machine(OS_WINDOWS, ARCH_ARM64, cores=10), libraries(ct2=False))
    assert arm.detector_backend == BACKEND_ONNX
    assert arm.detector is not None and arm.detector.provider == "cpu" and arm.detector.threads == 8
    assert any("Windows on ARM" in note for note in arm.notes)
    linux = make_plan(machine(OS_LINUX, ARCH_ARM64), libraries(ct2=False))
    assert any("not installed" in note for note in linux.notes)
    nothing = make_plan(machine(), libraries(ct2=False, sherpa=False))
    assert nothing.detector_backend == BACKEND_NONE and nothing.detector is None and nothing.parallel is False
    assert nothing.describe()[0].startswith("Detector: none available")


def test_plan_warns_on_low_memory_and_serialises() -> None:
    plan = make_plan(machine(memory=4.0), libraries())
    assert any("GB of memory" in note for note in plan.notes)
    doc = plan.to_dict()
    assert doc["detector_backend"] == BACKEND_CT2 and doc["publisher"]["provider"] == "cpu"
    assert doc["platform"] == "windows-x86_64" and isinstance(doc["notes"], list)
    assert doc["backends"] == {"ct2": True, "onnx": True} and doc["detector_onnx"]["provider"] == "cpu"
    assert make_plan(machine(), libraries(ct2=False, sherpa=False)).to_dict()["detector"] is None


def test_plan_carries_a_placement_per_backend() -> None:
    gpu = (Gpu(name="Test GPU", memory_mb=8192, driver="1.0"),)
    plan = make_plan(machine(), libraries(), gpus=gpu, cuda_count=1, ct2_types_cuda={"float16"})
    assert plan.placement_for(BACKEND_CT2) is not None and plan.placement_for(BACKEND_CT2).device == DEVICE_CUDA
    assert plan.placement_for(BACKEND_ONNX) is not None and plan.placement_for(BACKEND_ONNX).device == DEVICE_CPU
    assert plan.parallel_for(BACKEND_CT2) is True and plan.parallel_for(BACKEND_ONNX) is False
    assert plan.placement_for("other") is None and plan.parallel_for("other") is False
    onnx_only = make_plan(machine(), libraries(ct2=False))
    assert onnx_only.backends == {"ct2": False, "onnx": True} and onnx_only.placement_for(BACKEND_CT2) is None


def test_probes_return_documented_shapes() -> None:
    probed = hardware.probe_machine()
    assert probed.os in (OS_WINDOWS, OS_LINUX, OS_MACOS) or isinstance(probed.os, str)
    assert probed.arch in (ARCH_X86_64, ARCH_ARM64, ARCH_OTHER) and probed.cores >= 1
    libs = hardware.probe_libraries()
    assert isinstance(libs.sherpa_onnx, bool) and isinstance(libs.versions, dict)
    gpus = hardware.probe_nvidia_gpus()
    assert isinstance(gpus, tuple)
    assert hardware.cuda_device_count() >= 0
    assert isinstance(hardware.register_cuda_libraries(), tuple)
    plan = hardware.current_plan()
    assert plan.publisher.threads >= 1 and isinstance(plan.describe(), list)
    lines = describe_machine(probed, gpus, libs)
    assert lines[0].startswith("Machine:") and any(line.startswith("sherpa-onnx:") for line in lines)
