A project plan for **`jmlx`**—an idiomatic, zero-overhead Java 25+ framework that wraps Apple’s native `mlx-c` dynamic library using Project Panama (Foreign Function & Memory API).

---

### Architectural Layers

```
┌──────────────────────────────────────────────────────────────────┐
│                         User Application                         │
├──────────────────────────────────────────────────────────────────┤
│  se.alipsa.jmlx.nn      Layer, Linear, MultiHeadAttention, Loss  │  Idiomatic High-Level
├──────────────────────────────────────────────────────────────────┤  Java 25+ API
│  se.alipsa.jmlx.core    MLXArray, Stream, Device, Autograd       │
├──────────────────────────────────────────────────────────────────┤
│  se.alipsa.jmlx.memory  MLXScope, Cleaner Scopes                 │  Safe Off-Heap Bridge
├──────────────────────────────────────────────────────────────────┤
│  se.alipsa.jmlx.ffi     Generated jextract Bindings (mlx-c)      │  Raw Native Invocations
├──────────────────────────────────────────────────────────────────┤
│  libmlx.dylib / libmlxc.dylib                                    │  Apple Silicon Metal GPU
└──────────────────────────────────────────────────────────────────┘

```

---

### Core Design Principles

* **Zero-Copy Memory Interop:** Uses Java 25 `MemorySegment` to pass off-heap host memory directly to unified Apple Silicon GPU memory without buffer copies.
* **Automatic Native Memory Management:** Uses `java.lang.ref.Cleaner` attached to Java 25 `Arena` scopes to ensure `mlx_array_free` is triggered automatically when JVM objects are garbage collected.
* **Idiomatic Java Types:** Implements `AutoCloseable` on native scopes, exposes fluid functional APIs, and utilizes Java 25 record patterns for complex multi-array returns (e.g., `record SplitResult(MLXArray first, MLXArray second)`).
* **Lazy Computation Lifecycle:** Mirrors MLX’s lazy execution tree, executing operations only when `.eval()` or explicit array inspection methods (`.toArray()`, `.print()`) are called.

---

### Project Execution Plan

#### Phase 1: Native Binding Pipeline & Automation

* **Objective:** Establish automated FFM code generation over Apple’s `mlx-c` API.
* **Deliverables:**
* Set up CMake build script to build standard `libmlx.dylib` and `libmlx_c.dylib` target binaries.
* Configure `jextract` task inside Gradle/Maven to parse `mlx/c/*.h` headers into `se.alipsa.jmlx.ffi` internal package.
* Implement FFM linkage sanity tests verifying downcalls for core memory allocation and device queries (`mlx_get_default_device`).



#### Phase 2: Memory Management & `MLXArray` Wrapper

* **Objective:** Create a safe, leak-free Java handle around native `mlx_array` pointers.
* **Deliverables:**
* Build `MLXArray` class implementing `AutoCloseable`.
* Implement dual reference strategies:
* *Managed Scope:* Cleaned up automatically via JVM `Cleaner` when unreachable.
* *Confined Scope:* Bound to a explicit Java `Arena` for predictable memory disposal in tight inference loops.


* Implement type conversion utilities mapping primitive Java arrays (`float[]`, `int[]`, `MemorySegment`) directly into MLX native tensors.



#### Phase 3: Tensor Operations & Lazy Evaluation Engine

* **Objective:** Expose NumPy/PyTorch-style array manipulations.
* **Status:** Delivered. See `req/phase3-plan.md` for the detailed plan and decisions.
* **Deliverables:**
* **Element-wise Ops:** `add`, `subtract`, `multiply`, `divide`, `exp`, `log`, `sin`, `cos`.
* **Linear Algebra:** Matrix multiplication (`matmul`), transposed matrices, vector dot products, outer products.
  *Note: `mlx_dot` does not exist in mlx-c — it offers `mlx_inner`, `mlx_outer` and `mlx_tensordot` instead. NumPy's
  `dot` has rank-dependent semantics (vector dot for 1-D, matmul for 2-D, tensor contraction above); inventing a
  Java-side `MLX.dot` to reproduce them would have no native counterpart to defer to. This deliverable is satisfied
  by exposing mlx-c's own `inner`/`outer` — 1-D `inner` *is* the vector dot product — rather than by a `dot` method.*
* **Shape Manipulation:** `reshape`, `transpose`, `squeeze`, `broadcast_to`, `slice`.
* **Evaluation Engine:** ~~Thread-safe~~ **Thread-confined, enforced** `eval(MLXArray... arrays)` dispatchers
  triggering native evaluation on Apple Silicon GPUs.
  *Note: this substitutes the literal "thread-safe" deliverable above. `MLXScope` is confined by construction (see
  `req/initial-plan.md` §6), so a genuinely thread-safe `eval` would contradict the memory model the project already
  committed to. What was delivered instead is thread-confinement enforced at the API boundary: `eval` throws if
  called from a thread other than the one that owns the arrays' scope. This is a defensible reinterpretation, not
  the original wording — flagging it explicitly so it is not mistaken for the unmet literal deliverable.*



#### Phase 4: `se.alipsa.jmlx.nn` High-Level Neural Network Modules

* **Objective:** Port PyTorch-style layers for building Transformer models directly in Java.
* **Deliverables:**
* `Module` base class tracking trainable `MLXArray` weight parameters.
* Core neural network layers: `Linear`, `Embedding`, `RMSNorm`, `LayerNorm`, `ROPE` (Rotary Position Embeddings), `SiLU`, `GELU`.
* Quantized module support: `QuantizedLinear` handling low-bit weights (4-bit, 8-bit GGUF/MLX quantization).
* Attention utilities: `MultiHeadAttention` with KV-caching structures.



#### Phase 5: Model Loading & HuggingFace Interop

* **Objective:** Load pre-trained safetensors and GGUF model files into Java memory.
* **Deliverables:**
* Pure Java `.safetensors` parser converting raw memory segments directly into `MLXArray` tensors.
* Tokenizer integration (using Hugging Face `tokenizers` bindings or Java native tokenizers).
* `LlamaModel` and `QwenModel` reference implementations in Java demonstrating streaming LLM text generation.



---

### Code Architecture Example

```java
package se.alipsa.jmlx.example;

import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.Linear;

public class JMLXDemo {
    public static void main(String[] args) {
        // Explicit scope for zero-leak loop execution
        try (MLXScope scope = new MLXScope()) {
            // Allocate native MLX arrays directly from Java primitives
            MLXArray x = MLX.array(scope, new float[]{1.0f, 2.0f, 3.0f, 4.0f}, new int[]{2, 2});

            // Define a linear layer (2 inputs -> 4 outputs)
            Linear linear = new Linear(scope, 2, 4);

            // Forward pass (lazy evaluation)
            MLXArray y = linear.forward(x);

            // Materialize computation on Metal GPU
            MLX.eval(y);

            System.out.println("Result Shape: " + y.shape());
            System.out.println("Result Data: " + y.toFloatArray());
        } // Memory for x, y, and linear parameters instantly released here
    }
}

```

---

### Packaging & Distribution Strategy

| Target Artifact | Contents | Execution Environment |
| --- | --- | --- |
| `jmlx-core-25.jar` | Java classes, FFM abstractions, high-level `se.alipsa.jmlx.nn` | Any OS with Java 25+ |
| `jmlx-native-macos-arm64.jar` | Compiled `libmlx.dylib` & `libmlx_c.dylib` universal binaries | macOS (M1/M2/M3/M4 Apple Silicon) |

At runtime, `jmlx` automatically extracts and loads the matching `libmlx.dylib` native dynamic library from the classpath if no system-installed `mlx-c` library is found.
