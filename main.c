
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define INPUT_BITS 8
#define HIDDEN_NEURONS 4
#define BENCHMARK_RUNS 1000000

/*
 * Optimized weight storage:
 * 4 hidden neurons x 8 bits = 4 bytes
 * 1 output neuron x 4 bits = 1 byte
 *
 * Total = 5 bytes
 */

static const uint8_t hidden_weights[HIDDEN_NEURONS] = {
    0xA6, 0xF0, 0xCC, 0xAA
};

static const uint8_t output_weights = 0x0F;

volatile unsigned long benchmark_checksum = 0;

// Count the number of set bits
int count_bits(uint32_t number) {
    int count = 0;

    while (number != 0) {
        count += number & 1u;
        number >>= 1;
    }

    return count;
}

// XNOR-based binary dot product
int binary_dot_product(uint32_t input,
                       uint32_t weights,
                       int bits) {

    uint32_t mask = (bits == 32)
                  ? UINT32_MAX
                  : ((1u << bits) - 1u);

    uint32_t xnor = ~(input ^ weights) & mask;

    int matches = count_bits(xnor);

    return 2 * matches - bits;
}

// Binary activation function
int binary_activation(int value) {
    return value >= 0 ? 1 : -1;
}

// Validate and convert an 8-bit binary string
int parse_binary(const char *str, uint8_t *result) {

    if (strlen(str) != INPUT_BITS)
        return 0;

    uint8_t value = 0;

    for (int i = 0; i < INPUT_BITS; i++) {

        if (str[i] != '0' && str[i] != '1')
            return 0;

        value = (uint8_t)
                ((value << 1) | (str[i] - '0'));
    }

    *result = value;
    return 1;
}

// Optimized binary neural network
int binary_network(uint8_t input, int verbose) {

    // Four hidden activations packed into one byte
    uint8_t hidden_bits = 0;

    if (verbose)
        printf("\n--- HIDDEN LAYER ---\n");

    for (int i = 0; i < HIDDEN_NEURONS; i++) {

        int dot = binary_dot_product(
            input,
            hidden_weights[i],
            INPUT_BITS
        );

        int activation = binary_activation(dot);

        if (activation == 1) {
            hidden_bits |= (uint8_t)(1u << i);
        }

        if (verbose) {
            printf(
                "Neuron %d | Dot: %3d | Output: %+d\n",
                i + 1, dot, activation
            );
        }
    }

    int final_dot = binary_dot_product(
        hidden_bits,
        output_weights,
        HIDDEN_NEURONS
    );

    int prediction = binary_activation(final_dot);

    if (verbose) {

        printf("\n--- OUTPUT LAYER ---\n");

        printf("Packed Hidden Bits: ");

        for (int i = HIDDEN_NEURONS - 1; i >= 0; i--) {
            printf("%d", (hidden_bits >> i) & 1u);
        }

        printf("\nOutput Dot Product: %d\n",
               final_dot);

        printf("Final Prediction  : %+d\n",
               prediction);

        printf("Predicted Class   : %d\n",
               prediction == 1 ? 1 : 0);
    }

    return prediction;
}

// Float32 reference implementation
int float_network(uint8_t input) {

    float hidden[HIDDEN_NEURONS];

    for (int i = 0; i < HIDDEN_NEURONS; i++) {

        float dot = 0.0f;

        for (int j = 0; j < INPUT_BITS; j++) {

            float x = (input & (1u << j))
                    ? 1.0f : -1.0f;

            float w = (hidden_weights[i] & (1u << j))
                    ? 1.0f : -1.0f;

            dot += x * w;
        }

        hidden[i] = dot >= 0.0f ? 1.0f : -1.0f;
    }

    float output_dot = 0.0f;

    for (int i = 0; i < HIDDEN_NEURONS; i++) {

        float w = (output_weights & (1u << i))
                ? 1.0f : -1.0f;

        output_dot += hidden[i] * w;
    }

    return output_dot >= 0.0f ? 1 : -1;
}

// Compare weight and activation memory
void memory_comparison(void) {

    size_t total_weights =
        INPUT_BITS * HIDDEN_NEURONS
        + HIDDEN_NEURONS;

    size_t float_weight_memory =
        total_weights * sizeof(float);

    size_t binary_weight_memory =
        sizeof(hidden_weights)
        + sizeof(output_weights);

    size_t float_activation_memory =
        HIDDEN_NEURONS * sizeof(float);

    size_t binary_activation_memory =
        sizeof(uint8_t);

    double weight_saving =
        (1.0 - (double)binary_weight_memory /
        float_weight_memory) * 100.0;

    double activation_saving =
        (1.0 - (double)binary_activation_memory /
        float_activation_memory) * 100.0;

    printf("\n--- MEMORY COMPARISON ---\n");

    printf("Total Weights             : %zu\n",
           total_weights);

    printf("Float32 Weight Storage    : %zu bytes\n",
           float_weight_memory);

    printf("Binary Weight Storage     : %zu bytes\n",
           binary_weight_memory);

    printf("Weight Memory Saving      : %.2f%%\n",
           weight_saving);

    printf("\n--- ACTIVATION MEMORY ---\n");

    printf("Float32 Hidden Activations: %zu bytes\n",
           float_activation_memory);

    printf("Binary Hidden Activations : %zu bytes\n",
           binary_activation_memory);

    printf("Activation Memory Saving  : %.2f%%\n",
           activation_saving);
}

// Verify all 256 input patterns
void verify_networks(void) {

    int mismatches = 0;

    printf("\n--- CORRECTNESS VERIFICATION ---\n");

    for (int input = 0; input < 256; input++) {

        int binary_result =
            binary_network((uint8_t)input, 0);

        int float_result =
            float_network((uint8_t)input);

        if (binary_result != float_result) {
            mismatches++;
        }
    }

    printf("Patterns Tested : 256\n");
    printf("Mismatches      : %d\n", mismatches);

    printf("Status          : %s\n",
           mismatches == 0 ? "PASS" : "FAIL");
}

// Performance benchmarking
void performance_benchmark(void) {

    clock_t start, end;
    double binary_time, float_time;
    unsigned long checksum = 0;

    printf("\n--- PERFORMANCE BENCHMARK ---\n");

    printf("Iterations per Network: %d\n",
           BENCHMARK_RUNS);

    start = clock();

    for (int i = 0; i < BENCHMARK_RUNS; i++) {

        uint8_t input = (uint8_t)(i & 255);

        checksum += (unsigned long)
                    (binary_network(input, 0) + 1);
    }

    end = clock();

    binary_time =
        (double)(end - start) / CLOCKS_PER_SEC;

    benchmark_checksum = checksum;

    checksum = 0;

    start = clock();

    for (int i = 0; i < BENCHMARK_RUNS; i++) {

        uint8_t input = (uint8_t)(i & 255);

        checksum += (unsigned long)
                    (float_network(input) + 1);
    }

    end = clock();

    float_time =
        (double)(end - start) / CLOCKS_PER_SEC;

    benchmark_checksum += checksum;

    printf("Binary Network Time : %.6f seconds\n",
           binary_time);

    printf("Float32 Network Time: %.6f seconds\n",
           float_time);

    if (binary_time > 0.0) {
        printf("Float/Binary Ratio  : %.2fx\n",
               float_time / binary_time);
    }

    printf("Benchmark Completed!\n");
}

// Display architecture
void display_architecture(void) {

    printf("\n--- NETWORK ARCHITECTURE ---\n");

    printf("Input Layer  : %d neurons\n",
           INPUT_BITS);

    printf("Hidden Layer : %d binary neurons\n",
           HIDDEN_NEURONS);

    printf("Output Layer : 1 binary neuron\n");

    printf("\nArchitecture: 8 -> 4 -> 1\n");

    printf("Total Weights: 36\n");

    printf("Weight Encoding : +1 / -1\n");

    printf("Activation      : Binary Sign\n");

    printf("Operations      : XNOR + Popcount\n");

    printf("Weight Storage  : 5 bytes\n");
}

// Display terminal menu
void display_menu(void) {

    printf("\n========================================\n");
    printf("              BITNET-C v1.0\n");
    printf("       Binary Neural Network in C\n");
    printf("========================================\n");

    printf("\n             MAIN MENU\n\n");

    printf("  1. Run Binary Neural Network\n");
    printf("  2. Compare Memory Usage\n");
    printf("  3. Run Performance Benchmark\n");
    printf("  4. Verify Network Correctness\n");
    printf("  5. Display Network Architecture\n");
    printf("  6. Exit\n");

    printf("\n========================================\n");
    printf("Enter your choice: ");
}

int main(void) {

    int choice;
    char binary_input[100];
    uint8_t input;

    while (1) {

        display_menu();

        if (scanf("%d", &choice) != 1) {
            printf("\nInvalid menu input. Exiting.\n");
            break;
        }

        switch (choice) {

            case 1:

                printf("\nEnter an 8-bit binary pattern: ");

                if (scanf("%99s", binary_input) != 1) {
                    printf("Input error.\n");
                    return 1;
                }

                if (!parse_binary(binary_input, &input)) {
                    printf("Invalid binary input!\n");
                    break;
                }

                binary_network(input, 1);
                break;

            case 2:
                memory_comparison();
                break;

            case 3:
                performance_benchmark();
                break;

            case 4:
                verify_networks();
                break;

            case 5:
                display_architecture();
                break;

            case 6:
                printf("\nThank you for using BitNet-C!\n");
                return 0;

            default:
                printf("\nInvalid choice. Try again.\n");
        }
    }

    return 0;
}
