# Platforms and acceleration

twinscribe runs on the processor everywhere and uses an NVIDIA device where one is usable. On
start, and before every batch, the machine is probed and an acceleration plan is chosen: which
detector backend, on which device, with which compute type or provider, and whether the two
transcription engines can run at the same time. The plan is printed by `twinscribe check`,
shown in the settings, and recorded in every run record, so a run always states what it used.

## What runs where

| Platform | Publisher (transducer, sherpa-onnx) | Detector, CTranslate2 | Detector, ONNX (sherpa-onnx) | NVIDIA device | Window |
|---|---|---|---|---|---|
| Windows, x86-64 | wheel | wheel | wheel | CTranslate2 with the CUDA runtime packages; sherpa-onnx with its CUDA build | wheel |
| Windows, ARM64 | wheel | no wheel | wheel | none | wheel |
| Linux, x86-64 | wheel | wheel | wheel | as Windows x86-64 | wheel |
| Linux, aarch64 | wheel | wheel | wheel | as Windows x86-64 | wheel |
| macOS, Apple silicon and Intel | wheel | wheel | wheel | none; the processor path uses the platform's accelerated maths | wheel |

Wheel availability is as published by each project at the time of writing; the `engines`
extra installs faster-whisper only where CTranslate2 has a wheel, and sherpa-onnx everywhere.
The voice detector, the speaker models and the audio tagger that marks silence, music and
noise all run through sherpa-onnx on the processor on every platform in the table; the tagger
is small enough that no accelerator is ever used for it.

Checked on the development machine (Windows on ARM): the publisher, the speaker models and the
ONNX detector natively; the whole CTranslate2 path in an x86-64 environment under the platform's
emulation. Linux and macOS are covered by wheels and by the platform branches of the probes;
they have not been run.

## The plan

1. The detector backend is CTranslate2 when faster-whisper and CTranslate2 import, else the
   ONNX path through sherpa-onnx. A level's detector candidates are tried in that order, so a
   machine holding only the ONNX export of a model still offers the level.
2. A CUDA device is used when the preference allows it (`--device auto`, the default, or
   `cuda`) and CTranslate2 can see one; the detector then runs in float16 (or int8_float16
   where float16 is unsupported). On the processor the measured configuration, int8, is used.
3. The transducer and the speaker models use the CUDA provider only when the installed
   sherpa-onnx is its CUDA build (its version string says so); the processor build falls back
   with a note.
4. The two transcription engines run at the same time when they sit on different devices
   (typically the detector on the device and the publisher on the processor); otherwise one
   after the other, each with every core.
5. Threads per engine default to the smaller of eight and the core count, the ceiling the
   design found useful on laptop parts; the setting can raise it.
6. `--device cpu` (or Processor only in the settings) keeps every engine on the processor,
   for reproducibility or when a device misbehaves.

## Enabling a CUDA device

CTranslate2 needs the CUDA runtime libraries, cuBLAS and cuDNN, which no administrator install
is needed for: the `cuda` extra installs them as packages beside the libraries, and the
application registers their folders with the process before the engine loads. The transducer
and the speaker models need the CUDA build of sherpa-onnx, which the sherpa-onnx project
publishes on its own index rather than the package index; with the processor build they stay
on the processor and the plan says so.

## The ONNX detector, and what it costs

Whisper through sherpa-onnx decodes the recording in windows of up to thirty seconds, cut at
the quietest moment before each limit. The windows are deliberately not the voice detector's
utterances: the transducer decodes those, and a detector that saw the same boundaries failed
in the same places (measured: the review list found nothing). With its own windows the ONNX
turbo export transcribed the synthesised fixture as well as the CTranslate2 conversion did.

The published exports carry no cross-attention outputs, so they give segment timestamps but no
word timestamps. Words are placed inside each segment, confined to the parts the voice detector
marks as speech, so that no word lands in a pause. On the synthesised fixture this found three
of the five spans the CTranslate2 detector found, all three real, and no false alarm; the two
it missed were two-word gaps, which is what approximate word times cost. The transcript, the
text and the Word document say when a detector's word times are approximate.

An export made with the sherpa-onnx script that keeps the attention outputs
(`scripts/whisper/export-onnx-with-attention.py`, on a machine with PyTorch) drops into the same
model folder and gives token timestamps; the detector uses them without any change.

## Other accelerators

Integrated graphics through DirectML measured far slower than the processor for these models
and is not used. Apple's CoreML provider exists in sherpa-onnx and is not enabled here, because
its benefit for these models has not been measured. Dedicated neural processors are not used;
they need driver-level installation on the machines this tool is for.
