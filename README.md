# BitNet-C
### A Bitwise-Optimized Binary Neural Network for Ultra-Low Memory Activation

**BitNet-C** is a lightweight binary neural network inference prototype developed entirely in C. It explores how binary weight representation, packed activations, and bitwise operations can reduce neural-network storage requirements compared with conventional 32-bit floating-point implementations.

The project focuses on memory-efficient inference, computational optimization, and performance evaluation without relying on external machine-learning frameworks.

## Key Features

- **Pure C implementation:** Built using standard C libraries.
- **Binary neural network:** An 8–4–1 feedforward architecture.
- **XNOR-based computation:** Binary dot products using bitwise operations and population counting.
- **Compact weight storage:** Stores 36 binary weights in 5 bytes.
- **Packed activations:** Stores four hidden-neuron activations in one byte.
- **Interactive terminal interface:** Six menu options for predictions, comparisons, benchmarking, verification, and architecture inspection.
- **Float32 reference implementation:** Provides equivalent calculations for comparison.
- **Exhaustive input verification:** Compares both implementations across all 256 possible 8-bit input patterns.

## Network Architecture

```text
        INPUT LAYER
         8 Inputs
             |
             v
        HIDDEN LAYER
      4 Binary Neurons
             |
      XNOR + Popcount
             |
             v
        OUTPUT LAYER
       1 Binary Neuron
             |
             v
      Class 0 / Class 1
```

**Architecture:** 8 → 4 → 1

**Total connections:** 36

**Weight representation:** +1 / −1

**Activation function:** Binary sign function

## How It Works

1. The user enters an 8-bit binary pattern.
2. The program validates and converts the input into an unsigned 8-bit integer.
3. Four hidden neurons compute binary dot products using XNOR and population counting.
4. Each hidden neuron produces a +1 or −1 activation.
5. The four hidden activations are packed into a single byte.
6. The output neuron processes the packed activations and generates a binary classification.
7. The program can display memory comparisons, benchmark results, and correctness verification.

For binary vectors of length `N`, the dot product is calculated as:

```text
Binary Dot Product = 2 × Popcount(XNOR(A, B)) − N
```

## Memory Optimization

The network uses 36 binary weights and four hidden activations.

| Component | Float32 Storage | Binary Storage | Reduction |
|---|---:|---:|---:|
| Network weights | 144 bytes | 5 bytes | 96.53% |
| Hidden activations | 16 bytes | 1 byte | 93.75% |

These values describe the storage required for the specified weights and activation representations, not total application RAM consumption.

## Technologies Used

- C Programming Language
- GCC Compiler
- Standard C Libraries
- Bitwise Operations
- Binary Neural Networks
- OnlineGDB
- Git and GitHub

## Getting Started

### Requirements

A C compiler supporting C11, such as GCC.

No external machine-learning libraries are required.

### Compile

```bash
gcc -std=c11 -O2 main.c -o bitnet
```

### Run on Linux/macOS

```bash
./bitnet
```

### Run on Windows

Compile using:

```bash
gcc -std=c11 -O2 main.c -o bitnet.exe
```

Then execute:

```bash
.\bitnet.exe
```

Alternatively, open `main.c` in OnlineGDB and click **Run**.

## Interactive Menu

```text
========================================
              BITNET-C v1.0
       Binary Neural Network in C
========================================

             MAIN MENU

  1. Run Binary Neural Network
  2. Compare Memory Usage
  3. Run Performance Benchmark
  4. Verify Network Correctness
  5. Display Network Architecture
  6. Exit
```

## Example

**Input:**

```text
10110110
```

**Output:**

```text
Packed Hidden Bits: 1011
Output Dot Product: 2
Final Prediction  : +1
Predicted Class   : 1
```

## Correctness Verification

The program evaluates all 256 possible 8-bit inputs and compares predictions from the binary implementation with those from the float32 reference implementation.

Expected verification result:

```text
Patterns Tested : 256
Mismatches      : 0
Status          : PASS
```

This establishes computational agreement between the two implementations. It does not measure classification accuracy against a labeled dataset.

## Performance Benchmarking

The program runs 1,000,000 inference iterations for each implementation and reports their CPU execution times.

Results depend on compiler optimization, processor architecture, and runtime conditions. Binary computation is not guaranteed to outperform float32 computation in every environment.

## Current Limitations

- Uses predefined weights rather than a learned model.
- Does not implement backpropagation or neural-network training.
- Supports 8-bit input patterns.
- Does not include a labeled dataset for evaluating predictive accuracy.
- Memory comparisons focus on weight and activation storage, not peak process memory.
- The project is a BitNet-inspired binary neural-network prototype, not an implementation of Microsoft's BitNet architecture.

## Future Enhancements

- Add training and weight optimization.
- Implement labeled pattern-classification datasets.
- Support larger neural-network architectures.
- Add optimized hardware population-count instructions.
- Measure peak runtime memory consumption.
- Extend the network to embedded and resource-constrained devices.

## Project Lead & Developer
**Role: Team Lead, Lead Developer & Repository Maintainer**

**Nimit**  
B.Tech Computer Science and Engineering  
University of Petroleum and Energy Studies (UPES)

GitHub: [realnimit1-byte](https://github.com/realnimit1-byte)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
