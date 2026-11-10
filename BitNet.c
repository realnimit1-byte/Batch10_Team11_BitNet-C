
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define INPUT_BITS 8
#define HIDDEN_NEURONS 4
#define TOTAL_PATTERNS 256
#define TEST_SIZE 64
#define TRAIN_SIZE 192
#define BENCHMARK_RUNS 1000000
#define RESTARTS 12
#define MAX_PASSES 12
#define MODEL_FILE "bitnet_model.dat"

/*
 * BITNET-C v2.4
 * Binary Neural Network with Visualizer
 *
 * Architecture: 8 -> 4 -> 1
 * Binary weights: 5 bytes
 * Hidden activations: 1 byte
 */

static uint8_t hidden_weights[HIDDEN_NEURONS] = {
    0xA6, 0xF0, 0xCC, 0xAA
};

static uint8_t output_weights = 0x0F;

static const uint8_t original_hidden[HIDDEN_NEURONS] = {
    0xA6, 0xF0, 0xCC, 0xAA
};

static const uint8_t original_output = 0x0F;

static uint8_t test_mask[TOTAL_PATTERNS];
static int trained = 0;
static uint32_t random_state = 123456789u;

volatile unsigned long benchmark_checksum = 0;

/* ---------- BITWISE ENGINE ---------- */

int count_bits(uint32_t value) {
    int count = 0;

    while (value) {
        count += value & 1u;
        value >>= 1;
    }

    return count;
}

int binary_dot_product(uint32_t input,
                       uint32_t weights,
                       int bits) {

    uint32_t mask = (bits == 32)
                  ? UINT32_MAX
                  : ((1u << bits) - 1u);

    uint32_t matches = ~(input ^ weights) & mask;

    return 2 * count_bits(matches) - bits;
}

int binary_activation(int value) {
    return value >= 0 ? 1 : -1;
}

uint32_t next_random(void) {
    random_state =
        random_state * 1664525u + 1013904223u;

    return random_state;
}

/* ---------- INPUT UTILITIES ---------- */

int parse_binary(const char *text, uint8_t *result) {

    if (strlen(text) != INPUT_BITS)
        return 0;

    uint8_t value = 0;

    for (int i = 0; i < INPUT_BITS; i++) {

        if (text[i] != '0' && text[i] != '1')
            return 0;

        value = (uint8_t)(
            (value << 1) | (text[i] - '0')
        );
    }

    *result = value;
    return 1;
}

void print_binary(uint8_t value) {

    for (int i = 7; i >= 0; i--)
        printf("%d", (value >> i) & 1u);
}

void print_spaced_bits(uint8_t value, int bits) {

    for (int i = bits - 1; i >= 0; i--)
        printf("%d ", (value >> i) & 1u);
}

/* ---------- CLASSIFICATION LABEL ---------- */

int expected_class(uint8_t input) {
    return count_bits(input) >= 4 ? 1 : 0;
}

/* ---------- DATASET PREPARATION ---------- */

void shuffle_array(int arr[], int size) {

    for (int i = size - 1; i > 0; i--) {

        int j = (int)(
            next_random() % (uint32_t)(i + 1)
        );

        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

void prepare_dataset(void) {

    int class0[TOTAL_PATTERNS];
    int class1[TOTAL_PATTERNS];

    int n0 = 0;
    int n1 = 0;

    memset(test_mask, 0, sizeof(test_mask));

    random_state = 123456789u;

    for (int i = 0; i < TOTAL_PATTERNS; i++) {

        if (expected_class((uint8_t)i) == 0)
            class0[n0++] = i;
        else
            class1[n1++] = i;
    }

    shuffle_array(class0, n0);
    shuffle_array(class1, n1);

    for (int i = 0; i < 23; i++)
        test_mask[class0[i]] = 1;

    for (int i = 0; i < 41; i++)
        test_mask[class1[i]] = 1;
}

void display_dataset(void) {

    int train0 = 0, train1 = 0;
    int test0 = 0, test1 = 0;

    for (int i = 0; i < TOTAL_PATTERNS; i++) {

        int label = expected_class((uint8_t)i);

        if (test_mask[i]) {
            if (label == 0) test0++;
            else test1++;
        } else {
            if (label == 0) train0++;
            else train1++;
        }
    }

    printf("\n--- DATASET INFORMATION ---\n");

    printf("Total Patterns    : 256\n");
    printf("Training Patterns : 192\n");
    printf("Testing Patterns  : 64\n");

    printf("\nTraining Class 0  : %d\n", train0);
    printf("Training Class 1  : %d\n", train1);

    printf("Testing Class 0   : %d\n", test0);
    printf("Testing Class 1   : %d\n", test1);

    printf("\nSplit: Shuffled Stratified Holdout\n");
    printf("Test Fraction: 25%%\n");
    printf("Random Seed: 123456789\n");
}

/* ---------- BINARY NEURAL NETWORK ---------- */

int binary_network(uint8_t input, int verbose) {

    uint8_t hidden_bits = 0;

    if (verbose)
        printf("\n--- HIDDEN LAYER ---\n");

    for (int i = 0; i < HIDDEN_NEURONS; i++) {

        int dot = binary_dot_product(
            input, hidden_weights[i], INPUT_BITS
        );

        int activation = binary_activation(dot);

        if (activation == 1)
            hidden_bits |= (uint8_t)(1u << i);

        if (verbose) {
            printf(
                "Neuron %d | Dot: %3d | Output: %+d\n",
                i + 1, dot, activation
            );
        }
    }

    int final_dot = binary_dot_product(
        hidden_bits, output_weights, HIDDEN_NEURONS
    );

    int prediction = binary_activation(final_dot);

    if (verbose) {

        printf("\n--- OUTPUT LAYER ---\n");

        printf("Packed Hidden Bits: ");

        for (int i = HIDDEN_NEURONS - 1; i >= 0; i--)
            printf("%d", (hidden_bits >> i) & 1u);

        printf("\nOutput Dot Product: %d\n", final_dot);

        printf("Final Prediction  : %+d\n", prediction);

        printf("Predicted Class   : %d\n",
               prediction == 1 ? 1 : 0);
    }

    return prediction;
}

int predicted_class(uint8_t input) {
    return binary_network(input, 0) == 1 ? 1 : 0;
}

/* ---------- FLOAT32 REFERENCE ---------- */

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

    float dot = 0.0f;

    for (int i = 0; i < HIDDEN_NEURONS; i++) {

        float w = (output_weights & (1u << i))
                ? 1.0f : -1.0f;

        dot += hidden[i] * w;
    }

    return dot >= 0.0f ? 1 : -1;
}

/* ---------- ACCURACY CALCULATION ---------- */

int count_correct(int test_only) {

    int correct = 0;

    for (int i = 0; i < TOTAL_PATTERNS; i++) {

        if (test_only == 1 && !test_mask[i])
            continue;

        if (test_only == 0 && test_mask[i])
            continue;

        if (predicted_class((uint8_t)i)
            == expected_class((uint8_t)i))
            correct++;
    }

    return correct;
}

void evaluate_accuracy(void) {

    int train_correct = count_correct(0);
    int test_correct = count_correct(1);

    int tp = 0, tn = 0, fp = 0, fn = 0;

    for (int i = 0; i < TOTAL_PATTERNS; i++) {

        if (!test_mask[i])
            continue;

        int predicted = predicted_class((uint8_t)i);
        int expected = expected_class((uint8_t)i);

        if (predicted == 1 && expected == 1) tp++;
        else if (predicted == 0 && expected == 0) tn++;
        else if (predicted == 1) fp++;
        else fn++;
    }

    printf("\n--- CLASSIFICATION EVALUATION ---\n");

    printf("Model Status: %s\n",
           trained ? "Optimized/Loaded" : "Original");

    printf("\nTraining Correct  : %d / 192\n",
           train_correct);

    printf("Training Accuracy : %.2f%%\n",
           train_correct * 100.0 / TRAIN_SIZE);

    printf("\nTest Correct      : %d / 64\n",
           test_correct);

    printf("Test Accuracy     : %.2f%%\n",
           test_correct * 100.0 / TEST_SIZE);

    printf("\nOverall Correct   : %d / 256\n",
           train_correct + test_correct);

    printf("Overall Accuracy  : %.2f%%\n",
           (train_correct + test_correct) * 100.0
           / TOTAL_PATTERNS);

    printf("\n--- TEST SET CONFUSION MATRIX ---\n");

    printf("True Positives : %d\n", tp);
    printf("True Negatives : %d\n", tn);
    printf("False Positives: %d\n", fp);
    printf("False Negatives: %d\n", fn);
}

/* ---------- TRAINING ALGORITHM ---------- */

void train_network(void) {

    uint8_t best_hidden[HIDDEN_NEURONS];
    uint8_t best_output;

    memcpy(hidden_weights, original_hidden,
           sizeof(hidden_weights));

    output_weights = original_output;

    int initial_score = count_correct(0);
    int best_score = initial_score;

    memcpy(best_hidden, hidden_weights,
           sizeof(best_hidden));

    best_output = output_weights;

    printf("\n--- BINARY WEIGHT OPTIMIZATION ---\n");

    printf("Training Patterns : 192\n");
    printf("Held-out Test     : 64\n");
    printf("Random Restarts   : %d\n", RESTARTS);

    printf("\nInitial Training Accuracy: %.2f%%\n",
           initial_score * 100.0 / TRAIN_SIZE);

    random_state = 987654321u;

    for (int restart = 0; restart < RESTARTS; restart++) {

        if (restart == 0) {

            memcpy(hidden_weights, best_hidden,
                   sizeof(hidden_weights));

            output_weights = best_output;

        } else {

            for (int i = 0; i < HIDDEN_NEURONS; i++)
                hidden_weights[i] =
                    (uint8_t)(next_random() >> 24);

            output_weights =
                (uint8_t)(next_random() & 0x0F);
        }

        int score = count_correct(0);

        for (int pass = 0; pass < MAX_PASSES; pass++) {

            int improved = 0;

            for (int neuron = 0;
                 neuron < HIDDEN_NEURONS; neuron++) {

                for (int bit = 0;
                     bit < INPUT_BITS; bit++) {

                    hidden_weights[neuron] ^=
                        (uint8_t)(1u << bit);

                    int new_score = count_correct(0);

                    if (new_score > score) {
                        score = new_score;
                        improved = 1;
                    } else {
                        hidden_weights[neuron] ^=
                            (uint8_t)(1u << bit);
                    }
                }
            }

            for (int bit = 0;
                 bit < HIDDEN_NEURONS; bit++) {

                output_weights ^=
                    (uint8_t)(1u << bit);

                int new_score = count_correct(0);

                if (new_score > score) {
                    score = new_score;
                    improved = 1;
                } else {
                    output_weights ^=
                        (uint8_t)(1u << bit);
                }
            }

            if (!improved)
                break;
        }

        if (score > best_score) {

            best_score = score;

            memcpy(best_hidden, hidden_weights,
                   sizeof(best_hidden));

            best_output = output_weights;
        }
    }

    memcpy(hidden_weights, best_hidden,
           sizeof(hidden_weights));

    output_weights = best_output;
    trained = 1;

    printf("\nFinal Training Accuracy: %.2f%%\n",
           best_score * 100.0 / TRAIN_SIZE);

    printf("Training Correct: %d / 192\n", best_score);

    printf("\nOptimized Weights:\n");

    for (int i = 0; i < HIDDEN_NEURONS; i++)
        printf("Neuron %d: 0x%02X\n",
               i + 1, hidden_weights[i]);

    printf("Output: 0x%02X\n", output_weights);

    printf("\nOptimization Completed!\n");
}

/* ---------- CUSTOM PATTERN ---------- */

void test_pattern(void) {

    char text[100];
    uint8_t input;

    printf("\nEnter an 8-bit binary pattern: ");

    if (scanf("%99s", text) != 1) {
        printf("Input error.\n");
        return;
    }

    if (!parse_binary(text, &input)) {
        printf("Invalid binary input!\n");
        return;
    }

    printf("\n--- PATTERN RECOGNITION ---\n");

    printf("Input: ");
    print_binary(input);

    printf("\nNumber of 1s: %d\n", count_bits(input));

    int prediction = binary_network(input, 1);
    int expected = expected_class(input);

    printf("\nExpected Class : %d\n", expected);
    printf("Predicted Class: %d\n",
           prediction == 1 ? 1 : 0);

    printf("Result: %s\n",
           (prediction == 1 ? 1 : 0) == expected
           ? "CORRECT" : "INCORRECT");
}

/* ---------- MEMORY COMPARISON ---------- */

void memory_comparison(void) {

    size_t total_weights =
        INPUT_BITS * HIDDEN_NEURONS + HIDDEN_NEURONS;

    size_t float_weights =
        total_weights * sizeof(float);

    size_t binary_weights =
        sizeof(hidden_weights) + sizeof(output_weights);

    size_t float_activations =
        HIDDEN_NEURONS * sizeof(float);

    size_t binary_activations = sizeof(uint8_t);

    printf("\n--- MEMORY COMPARISON ---\n");

    printf("Total Weights          : %zu\n",
           total_weights);

    printf("Float32 Weight Storage : %zu bytes\n",
           float_weights);

    printf("Binary Weight Storage  : %zu bytes\n",
           binary_weights);

    printf("Weight Saving          : %.2f%%\n",
           (1.0 - (double)binary_weights /
           float_weights) * 100.0);

    printf("\nFloat32 Activations    : %zu bytes\n",
           float_activations);

    printf("Binary Activations     : %zu bytes\n",
           binary_activations);

    printf("Activation Saving      : %.2f%%\n",
           (1.0 - (double)binary_activations /
           float_activations) * 100.0);

    printf("\nNote: Storage excludes program overhead.\n");
}

/* ---------- CORRECTNESS VERIFICATION ---------- */

void verify_networks(void) {

    int mismatches = 0;

    for (int i = 0; i < TOTAL_PATTERNS; i++) {

        uint8_t input = (uint8_t)i;

        if (binary_network(input, 0)
            != float_network(input))
            mismatches++;
    }

    printf("\n--- CORRECTNESS VERIFICATION ---\n");

    printf("Patterns Tested : 256\n");
    printf("Mismatches      : %d\n", mismatches);

    printf("Status          : %s\n",
           mismatches == 0 ? "PASS" : "FAIL");
}

/* ---------- PERFORMANCE BENCHMARK ---------- */

void performance_benchmark(void) {

    clock_t start, end;
    unsigned long checksum = 0;

    start = clock();

    for (int i = 0; i < BENCHMARK_RUNS; i++)
        checksum += (unsigned long)
            (binary_network((uint8_t)i, 0) + 1);

    end = clock();

    double binary_time =
        (double)(end - start) / CLOCKS_PER_SEC;

    benchmark_checksum = checksum;
    checksum = 0;

    start = clock();

    for (int i = 0; i < BENCHMARK_RUNS; i++)
        checksum += (unsigned long)
            (float_network((uint8_t)i) + 1);

    end = clock();

    double float_time =
        (double)(end - start) / CLOCKS_PER_SEC;

    benchmark_checksum += checksum;

    printf("\n--- PERFORMANCE BENCHMARK ---\n");

    printf("Iterations: %d per network\n",
           BENCHMARK_RUNS);

    printf("Binary Time : %.6f seconds\n", binary_time);
    printf("Float32 Time: %.6f seconds\n", float_time);

    if (binary_time > 0.0)
        printf("Float/Binary Ratio: %.2fx\n",
               float_time / binary_time);

    printf("Benchmark Completed!\n");
}

/* ---------- NETWORK ARCHITECTURE ---------- */

void display_architecture(void) {

    printf("\n--- NETWORK ARCHITECTURE ---\n");

    printf("Architecture: 8 -> 4 -> 1\n");
    printf("Total Weights: 36\n");

    printf("Binary Weight Storage: 5 bytes\n");
    printf("Hidden Activation Storage: 1 byte\n");

    printf("\nOperations: XNOR + Popcount\n");
    printf("Activation: Binary Sign\n");

    printf("\nTraining: Bit-Flip Hill Climbing\n");
    printf("Evaluation: Stratified Holdout\n");

    printf("\nClass 1: At least four 1-bits\n");
    printf("Class 0: Fewer than four 1-bits\n");

    printf("\nModel: %s\n",
           trained ? "Optimized/Loaded" : "Original");
}

/* ---------- MODEL FILE HANDLING ---------- */

int save_model(void) {

    FILE *file = fopen(MODEL_FILE, "wb");

    if (file == NULL) {
        printf("\nError: Unable to create model file.\n");
        return 0;
    }

    const unsigned char signature[4] = {
        'B', 'N', 'C', '1'
    };

    int success =
        fwrite(signature, 1, 4, file) == 4 &&
        fwrite(hidden_weights, 1,
               HIDDEN_NEURONS, file) == HIDDEN_NEURONS &&
        fwrite(&output_weights, 1, 1, file) == 1;

    if (fclose(file) != 0)
        success = 0;

    if (!success) {
        printf("\nError: Model could not be saved.\n");
        return 0;
    }

    printf("\n--- MODEL SAVED ---\n");
    printf("Filename : %s\n", MODEL_FILE);
    printf("Weights  : 5 bytes\n");
    printf("File Size: 9 bytes\n");
    printf("Status   : SUCCESS\n");

    return 1;
}

int load_model(void) {

    FILE *file = fopen(MODEL_FILE, "rb");

    if (file == NULL) {
        printf("\nError: Saved model file not found.\n");
        return 0;
    }

    unsigned char signature[4];
    uint8_t loaded_hidden[HIDDEN_NEURONS];
    uint8_t loaded_output;

    int success =
        fread(signature, 1, 4, file) == 4 &&
        memcmp(signature, "BNC1", 4) == 0 &&
        fread(loaded_hidden, 1,
              HIDDEN_NEURONS, file) == HIDDEN_NEURONS &&
        fread(&loaded_output, 1, 1, file) == 1;

    if (success) {
        if (loaded_output > 0x0F || fgetc(file) != EOF)
            success = 0;
    }

    if (fclose(file) != 0)
        success = 0;

    if (!success) {
        printf("\nError: Invalid or corrupted model file.\n");
        return 0;
    }

    memcpy(hidden_weights, loaded_hidden,
           sizeof(loaded_hidden));

    output_weights = loaded_output;
    trained = 1;

    printf("\n--- MODEL LOADED ---\n");
    printf("Filename : %s\n", MODEL_FILE);
    printf("Status   : SUCCESS\n");
    printf("Weights restored successfully.\n");

    return 1;
}

/* ---------- NEURAL NETWORK VISUALIZER ---------- */

void visualize_network(void) {

    char text[100];
    uint8_t input;

    printf("\nEnter an 8-bit binary pattern: ");

    if (scanf("%99s", text) != 1) {
        printf("Input error.\n");
        return;
    }

    if (!parse_binary(text, &input)) {
        printf("Invalid binary input!\n");
        return;
    }

    uint8_t hidden_bits = 0;

    printf("\n========================================\n");
    printf("       BITNET-C NEURAL VISUALIZER\n");
    printf("========================================\n");

    printf("\n[ INPUT LAYER - 8 NEURONS ]\n\n");

    printf("  ");
    print_spaced_bits(input, INPUT_BITS);

    printf("\n\n              |\n");
    printf("              v\n");

    printf("\n[ HIDDEN LAYER - 4 NEURONS ]\n\n");

    for (int i = 0; i < HIDDEN_NEURONS; i++) {

        uint8_t weight = hidden_weights[i];

        uint8_t xnor = (uint8_t)
            (~(input ^ weight));

        int matches = count_bits(xnor);

        int dot = 2 * matches - INPUT_BITS;

        int activation = binary_activation(dot);

        if (activation == 1)
            hidden_bits |= (uint8_t)(1u << i);

        printf("  Neuron %d\n", i + 1);

        printf("    Weights : ");
        print_binary(weight);

        printf("\n    XNOR    : ");
        print_binary(xnor);

        printf("\n    Matches : %d / 8\n", matches);
        printf("    Dot     : %+d\n", dot);
        printf("    Output  : %+d\n\n", activation);
    }

    printf("              |\n");
    printf("              v\n");

    printf("\n[ PACKED ACTIVATIONS ]\n\n");

    printf("  Hidden Bits: ");

    for (int i = HIDDEN_NEURONS - 1; i >= 0; i--)
        printf("%d", (hidden_bits >> i) & 1u);

    printf("\n  Storage: 1 byte\n");

    int final_dot = binary_dot_product(
        hidden_bits, output_weights, HIDDEN_NEURONS
    );

    int prediction = binary_activation(final_dot);

    printf("\n              |\n");
    printf("              v\n");

    printf("\n[ OUTPUT LAYER ]\n\n");

    printf("  Output Weights: ");

    for (int i = HIDDEN_NEURONS - 1; i >= 0; i--)
        printf("%d", (output_weights >> i) & 1u);

    printf("\n  Dot Product   : %+d\n", final_dot);

    printf("  Prediction    : Class %d\n",
           prediction == 1 ? 1 : 0);

    printf("\n[ CLASSIFICATION CHECK ]\n\n");

    int expected = expected_class(input);
    int predicted = prediction == 1 ? 1 : 0;

    printf("  Expected Class : %d\n", expected);
    printf("  Predicted Class: %d\n", predicted);

    printf("  Result         : %s\n",
           expected == predicted ? "CORRECT" : "INCORRECT");

    printf("\n========================================\n");
    printf("         VISUALIZATION COMPLETE\n");
    printf("========================================\n");
}

/* ---------- MAIN MENU ---------- */

void display_menu(void) {

    printf("\n========================================\n");
    printf("              BITNET-C v2.4\n");
    printf("     Binary Neural Network in C\n");
    printf("========================================\n");

    printf("\n              MAIN MENU\n\n");

    printf("  1. Run Binary Pattern Recognition\n");
    printf("  2. Compare Memory Usage\n");
    printf("  3. Run Performance Benchmark\n");
    printf("  4. Verify Network Correctness\n");
    printf("  5. Display Network Architecture\n");
    printf("  6. Train / Optimize Binary Weights\n");
    printf("  7. Evaluate Classification Accuracy\n");
    printf("  8. Display Dataset Information\n");
    printf("  9. Save Optimized Model\n");
    printf(" 10. Load Saved Model\n");
    printf(" 11. Neural Network Visualizer\n");
    printf(" 12. Exit\n");

    printf("\n========================================\n");
    printf("Enter your choice: ");
}

/* ---------- MAIN FUNCTION ---------- */

int main(void) {

    int choice;

    prepare_dataset();

    while (1) {

        display_menu();

        if (scanf("%d", &choice) != 1) {
            printf("\nInvalid menu input.\n");
            return 1;
        }

        switch (choice) {

            case 1:
                test_pattern();
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
                train_network();
                break;

            case 7:
                evaluate_accuracy();
                break;

            case 8:
                display_dataset();
                break;

            case 9:
                if (!trained) {
                    printf(
                        "\nPlease train or load a model first.\n"
                    );
                } else {
                    save_model();
                }
                break;

            case 10:
                load_model();
                break;

            case 11:
                visualize_network();
                break;

            case 12:
                printf(
                    "\nThank you for using BitNet-C!\n"
                );
                return 0;

            default:
                printf("\nInvalid choice. Try again.\n");
        }
    }

    return 0;
}
