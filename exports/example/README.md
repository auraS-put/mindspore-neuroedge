# Seizure Prediction Model — Deployment Guide

## Quick Start for Mobile App Developers

### Model File

```
exports/example/ultralight_cnn.ms    (19 KB)
```

This is a **MindSpore Lite** model ready for on-device inference on HarmonyOS
via the MindSpore Lite Kit.

### Model Specification

| Property | Value |
|---|---|
| Architecture | UltraLightCNN (depthwise-separable CNN + channel attention) |
| Parameters | 2,038 |
| Input shape | `float32[1, 4, 1024]` — (batch, channels, samples) |
| Output shape | `float32[1, 2]` — (batch, [interictal, preictal]) |
| Input meaning | 4 EEG channels × 1024 samples (4 seconds at 256 Hz) |
| Output meaning | Raw logits; apply sigmoid for probability |
| Latency (x86 CPU) | 0.17 ms per inference |
| Target device | Muse 2 / Muse S EEG headband |

### Channel Mapping

The model expects 4 EEG channels in this order:

| Index | Training (Siena) | Muse 2/S Equivalent |
|---|---|---|
| 0 | F7 | AF7 (left forehead) |
| 1 | F8 | AF8 (right forehead) |
| 2 | T7 | TP9 (left ear) |
| 3 | T8 | TP10 (right ear) |

### Preprocessing on Device

Before feeding data to the model, apply these steps to each 4-second window:

```
1. Buffer raw EEG:     Collect 4 × 1024 float32 values (4 channels, 4 seconds at 256 Hz)
2. Notch filter:       Remove power-line noise (50 Hz for EU, 60 Hz for US)
3. Bandpass filter:    Keep 0.5–45 Hz (2nd-order Butterworth IIR)
4. Z-score normalize:  For each channel: (x - mean) / std  (computed over the 1024 samples)
5. Pack as tensor:     float32[1, 4, 1024] in NCHW-like layout (batch=1, channels=4, time=1024)
```

### Inference Stride

Slide the window by **256 samples (1 second)** for 75% overlap, matching the
training configuration. This gives one prediction per second.

### Interpreting Output

```
output = model.predict(input)        // float32[1, 2]
prob_preictal = sigmoid(output[0][1]) // probability of upcoming seizure
if prob_preictal > 0.5:
    trigger_alert()
```

---

## Integration in DevEco Studio (HarmonyOS)

### Option A: ArkTS (recommended for prototyping)

```typescript
import { mindSporeLite } from '@kit.MindSporeLiteKit';

// 1. Load model from rawfile
let context: mindSporeLite.Context = { target: ['cpu'], cpu: { threadNum: 2, precisionMode: 'enforce_fp32' } };
let model = await mindSporeLite.loadModelFromFile('/path/to/ultralight_cnn.ms', context);

// 2. Set input
let inputs = model.getInputs();
inputs[0].setData(preprocessedEegBuffer);  // ArrayBuffer of 4×1024 float32

// 3. Run inference
let outputs = await model.predict(inputs);
let result = new Float32Array(outputs[0].getData());
// result[0] = interictal logit, result[1] = preictal logit
```

### Option B: C/C++ via N-API (lower latency)

```cpp
#include <mindspore/model.h>
#include <mindspore/context.h>
#include <mindspore/tensor.h>

// Build model
auto context = OH_AI_ContextCreate();
auto cpu_info = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
OH_AI_DeviceInfoSetEnableFP16(cpu_info, true);
OH_AI_ContextAddDeviceInfo(context, cpu_info);

auto model = OH_AI_ModelCreate();
OH_AI_ModelBuild(model, modelBuffer, modelSize, OH_AI_MODELTYPE_MINDIR, context);

// Set input data
auto inputs = OH_AI_ModelGetInputs(model);
float *data = (float *)OH_AI_TensorGetMutableData(inputs.handle_list[0]);
// Copy 4×1024 float32 preprocessed EEG into data[]

// Predict
auto outputs = OH_AI_ModelGetOutputs(model);
OH_AI_ModelPredict(model, inputs, &outputs, nullptr, nullptr);
float *out = (float *)OH_AI_TensorGetData(outputs.handle_list[0]);
// out[0] = interictal logit, out[1] = preictal logit
```

### Project Setup

1. Place `ultralight_cnn.ms` in `entry/src/main/resources/rawfile/`
2. Add `SystemCapability.AI.MindSporeLite` to `syscap.json` (ArkTS only)
3. Link `mindspore_lite_ndk` in `CMakeLists.txt` (C++ only)
4. DevEco Studio 4.1+, HarmonyOS SDK API 11+

---

## How This Model Was Produced

```
.ckpt (train) ──► .mindir (ms.export) ──► .ms (converter_lite)
  9 KB               34 KB                  19 KB
```

1. **Train**: MindSpore 2.8.0, `ultralight_cnn` config, 4-channel Siena EEG data
2. **Export**: `ms.export(model, dummy_input, file_format="MINDIR")`
3. **Convert**: `converter_lite --fmk=MINDIR --modelFile=*.mindir --outputFile=*`
   (MindSpore Lite 2.6.0, `mindspore-lite-2.6.0-linux-x64.tar.gz`)

### Reproducing the conversion

```bash
# Download converter_lite
wget https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.6.0/MindSpore/lite/release/linux/x86_64/mindspore-lite-2.6.0-linux-x64.tar.gz
tar xzf mindspore-lite-2.6.0-linux-x64.tar.gz

# Set environment
export MSLITE=mindspore-lite-2.6.0-linux-x64
export LD_LIBRARY_PATH=$MSLITE/tools/converter/lib:$MSLITE/runtime/lib

# Convert
$MSLITE/tools/converter/converter/converter_lite \
  --fmk=MINDIR \
  --modelFile=exports/deployment_test/ultralight_cnn.mindir \
  --outputFile=exports/example/ultralight_cnn
```

### Note on this example model

This `.ms` was exported from a model trained on **synthetic data** (1 epoch,
100 samples) for pipeline validation only. It demonstrates the correct input/output
shapes and proves the full `.ckpt → .mindir → .ms` conversion works.

For production use, retrain with the full Siena dataset and validate performance
metrics (sensitivity, specificity, AUC) before deployment.
