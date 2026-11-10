# BitNet-C v2.4 — Algorithm and Pseudocode

## 1. Introduction

BitNet-C is a bitwise-optimized binary neural network implemented in C. It demonstrates memory-efficient neural-network inference, binary-weight optimization, classification, and performance evaluation.

The network uses an **8 → 4 → 1 architecture**, consisting of eight input neurons, four hidden neurons, and one output neuron.

Unlike conventional floating-point neural networks, BitNet-C represents its weights using binary values and performs binary dot products using XNOR and population counting.

## 2. Problem Statement

Conventional neural networks commonly store weights and activations as 32-bit floating-point numbers, increasing memory requirements.

BitNet-C investigates whether binary weight representation and bitwise operations can reduce storage requirements while maintaining correct predictions on a defined binary-pattern classification task.

## 3. Classification Objective

The system classifies an 8-bit binary pattern according to the number of set bits.

| Condition | Classification |
|---|---|
| Fewer than four `1` bits | Class 0 |
| Four or more `1` bits | Class 1 |

For example:

- `00000000` → Class 0
- `00000111` → Class 0
- `10110110` → Class 1
- `11111111` → Class 1

The expected label is calculated separately from neural-network inference and is used for training and evaluation.

## 4. Network Architecture

```text
        INPUT LAYER
          8 Bits
             |
             v
      +---------------+
      | HIDDEN LAYER  |
      |  4 Neurons    |
      +---------------+
             |
       XNOR + Popcount
             |
       Binary Activation
             |
             v
      Packed Hidden Bits
           1 Byte
             |
             v
      +---------------+
      | OUTPUT LAYER  |
      |   1 Neuron    |
      +---------------+
             |
             v
       Class 0 / Class 1
```

**Total weights:** 36

**Binary weight storage:** 5 bytes

**Hidden activation storage:** 1 byte

## 5. Binary Dot Product Algorithm

The binary dot product is calculated using XNOR and population counting.

**Formula:**

```text
Binary Dot Product = 2 × Popcount(XNOR(A, B)) − N
```

Where:

- `A` represents the binary input.
- `B` represents the binary weights.
- `N` is the number of valid bits.
- `Popcount` counts the number of set bits.

### Pseudocode

```text
FUNCTION BinaryDotProduct(input, weights, bits)

    mask = (1 << bits) - 1

    matches = NOT(input XOR weights) AND mask

    count = Popcount(matches)

    dot_product = 2 * count - bits

    RETURN dot_product

END FUNCTION
```

For 32-bit operations, the implementation uses an appropriate full-width mask to avoid shifting by the integer width.

## 6. Binary Activation Algorithm

Each neuron produces either +1 or −1.

### Pseudocode

```text
FUNCTION BinaryActivation(value)

    IF value >= 0 THEN
        RETURN +1
    ELSE
        RETURN -1
    END IF

END FUNCTION
```

Positive activations are represented as `1` bits, while negative activations are represented as `0` bits when packed.

## 7. Neural Network Inference Algorithm

### Pseudocode

```text
FUNCTION BinaryNetwork(input)

    hidden_bits = 0

    FOR neuron = 0 TO 3

        dot = BinaryDotProduct(
            input,
            hidden_weights[neuron],
            8
        )

        activation = BinaryActivation(dot)

        IF activation == +1 THEN
            SET bit neuron in hidden_bits
        END IF

    END FOR

    output_dot = BinaryDotProduct(
        hidden_bits,
        output_weights,
        4
    )

    prediction = BinaryActivation(output_dot)

    RETURN prediction

END FUNCTION
```

The returned value is mapped to Class 1 for +1 and Class 0 for −1.

## 8. Dataset Preparation Algorithm

The program enumerates all 256 possible 8-bit inputs.

Expected labels are generated using the classification rule.

A deterministic shuffled, stratified holdout is used to divide the dataset into training and testing partitions.

### Dataset Distribution

| Category | Training | Testing |
|---|---:|---:|
| Class 0 | 70 | 23 |
| Class 1 | 122 | 41 |
| Total | 192 | 64 |

### Pseudocode

```text
FUNCTION PrepareDataset()

    Generate all inputs from 0 to 255

    Divide inputs into Class 0 and Class 1

    Shuffle both groups using a fixed random seed

    Select 23 Class 0 patterns for testing

    Select 41 Class 1 patterns for testing

    Assign remaining 192 patterns to training

END FUNCTION
```

The fixed seed makes the split reproducible.

## 9. Binary Weight Optimization Algorithm

BitNet-C uses bit-flip hill climbing with multiple random restarts.

The objective is to maximize the number of correct classifications on the training partition.

### Pseudocode

```text
FUNCTION TrainNetwork()

    Restore original weights

    best_score = TrainingAccuracy(original_weights)

    best_weights = original_weights

    FOR each restart

        IF first restart THEN
            Use best_weights
        ELSE
            Generate random binary weights
        END IF

        current_score = TrainingAccuracy(current_weights)

        FOR each optimization pass

            improved = FALSE

            FOR each hidden weight bit

                Flip the bit

                new_score = TrainingAccuracy(current_weights)

                IF new_score > current_score THEN
                    Keep the changed bit
                    current_score = new_score
                    improved = TRUE
                ELSE
                    Restore the bit
                END IF

            END FOR

            FOR each output weight bit

                Flip the bit

                Evaluate training score

                Keep the change only if score improves

            END FOR

            IF no improvement THEN
                STOP current restart
            END IF

        END FOR

        IF current_score > best_score THEN
            Save current weights as best_weights
        END IF

    END FOR

    Restore best_weights

END FUNCTION
```

**Training method:** Discrete bit-flip optimization, not gradient-based backpropagation.

The test partition is excluded from weight optimization.

## 10. Classification Accuracy Algorithm

The program evaluates predictions against expected labels.

### Formula

```text
Accuracy (%) = (Correct Predictions / Total Predictions) × 100
```

### Pseudocode

```text
FUNCTION EvaluateAccuracy()

    train_correct = 0
    test_correct = 0

    FOR each input from 0 to 255

        predicted = BinaryNetwork(input)

        expected = ExpectedClass(input)

        IF predicted == expected THEN

            IF input belongs to training partition THEN
                train_correct++
            ELSE
                test_correct++
            END IF

        END IF

    END FOR

    Calculate training accuracy

    Calculate test accuracy

    Calculate overall accuracy

    Display confusion matrix for test partition

END FUNCTION
```

## 11. Memory Optimization Algorithm

The network contains 36 weights.

### Weight Storage Comparison

| Representation | Storage |
|---|---:|
| Float32 weights | 144 bytes |
| Binary weights | 5 bytes |
| Storage reduction | 96.53% |

### Hidden Activation Comparison

| Representation | Storage |
|---|---:|
| Float32 activations | 16 bytes |
| Packed binary activations | 1 byte |
| Storage reduction | 93.75% |

The calculations compare the specified representations rather than total process RAM usage.

## 12. Model Saving Algorithm

The trained network can save its optimized weights to `bitnet_model.dat`.

### File Format

```text
+------------------------+
| Signature: BNC1        | 4 bytes
+------------------------+
| Hidden Neuron Weights  | 4 bytes
+------------------------+
| Output Neuron Weights  | 1 byte
+------------------------+

Total File Size: 9 bytes
```

### Pseudocode

```text
FUNCTION SaveModel()

    Open model file in binary write mode

    IF file cannot be opened THEN
        Display error
        RETURN failure
    END IF

    Write signature "BNC1"

    Write four hidden weight bytes

    Write one output weight byte

    Close file

    Display save status

END FUNCTION
```

## 13. Model Loading Algorithm

### Pseudocode

```text
FUNCTION LoadModel()

    Open model file in binary read mode

    IF file does not exist THEN
        Display error
        RETURN failure
    END IF

    Read and validate signature

    Read hidden and output weights

    Validate file length and output weight range

    IF validation succeeds THEN
        Restore network weights
        Mark model as loaded
    ELSE
        Reject the file
    END IF

    Close file

END FUNCTION
```

## 14. Performance Benchmarking Algorithm

The program compares binary and float32 inference performance.

### Pseudocode

```text
FUNCTION Benchmark()

    Start CPU timer

    Run binary inference 1,000,000 times

    Record binary execution time

    Start CPU timer

    Run float32 inference 1,000,000 times

    Record float32 execution time

    Calculate float_time / binary_time

    Display execution times and ratio

END FUNCTION
```

Execution times depend on the compiler, optimization settings, and runtime environment.

## 15. Neural Network Visualization Algorithm

The terminal visualizer demonstrates each inference stage.

### Pseudocode

```text
FUNCTION VisualizeNetwork()

    Read and validate an 8-bit input

    Display input bits

    FOR each hidden neuron

        Display binary weights

        Calculate and display XNOR result

        Display matching-bit count

        Calculate binary dot product

        Display activation

    END FOR

    Display packed hidden activations

    Calculate output neuron result

    Display predicted class

    Compare prediction with expected class

END FUNCTION
```

## 16. Main Program Algorithm

### Pseudocode

```text
START

Prepare training and testing dataset

REPEAT

    Display main menu

    Read user choice

    SWITCH choice

        1: Run binary pattern recognition

        2: Display memory comparison

        3: Run performance benchmark

        4: Verify binary and float32 predictions

        5: Display network architecture

        6: Train binary weights

        7: Evaluate classification accuracy

        8: Display dataset information

        9: Save optimized model

       10: Load saved model

       11: Display neural network visualizer

       12: Exit program

        DEFAULT: Display invalid choice

    END SWITCH

UNTIL user selects Exit

STOP
```

## 17. C Programming Concepts Used

| C Concept | Application in BitNet-C |
|---|---|
| Functions | Modular implementation of neural-network operations |
| Arrays | Weight storage and dataset organization |
| Loops | Training, inference, and evaluation |
| Conditional statements | Activation and classification logic |
| Bitwise operators | XOR, XNOR, shifting, masking |
| Fixed-width integers | `uint8_t`, `uint32_t` |
| Pointers | Input parsing and file operations |
| File handling | Model saving and loading |
| Structures of control | Menu-driven application |
| Standard libraries | Input/output, strings, timing |

## 18. Conclusion

BitNet-C demonstrates how binary representations and bitwise operations can be applied to neural-network inference in C.

The project combines compact weight storage, packed hidden activations, discrete weight optimization, classification evaluation, file handling, and interactive visualization.

Its implementation provides a practical demonstration of C programming concepts within a small, memory-efficient neural-network architecture.

---

## Project Information

**Project:** BitNet-C v2.4

**Institution:** UPES, Dehradun

**Batch:** 10

**Team:** Amateurs

**Team Lead:** Nimit

**Team Members:**
1. Nimit — Team Lead
2. Abhimanyu Jakhar
3. Aasma Sharda
4. Pranav Bhasin

**Repository:** [BitNet-C](https://github.com/realnimit1-byte/BitNet-C)

**© 2026 BitNet-C Project Team | Maintained by Nimit**
