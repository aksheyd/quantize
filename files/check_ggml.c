// The C half of `just ggml-check`. It reads quantized_from_rust.gguf, which
// check_gguf.rs wrote, with ggml, the library inside llama.cpp, and runs it
// as llama.cpp runs a model on the CPU. Next to each Q4_0 or Q8_0 tensor
// NAME, the file holds NAME.decoded, the values quantize decodes it to, and
// NAME.product, the product of quantize's matmul with the 16 vectors in
// `inputs`.
//
// 1. ggml's own gguf reader loads the file.
// 2. ggml's to_float decodes each tensor to NAME.decoded, bit for bit.
// 3. ggml's CPU backend multiplies each tensor by the first input, then by
//    all 16, which run different kernels. It does both from the plain CPU
//    buffer, and again from the CPU_REPACK buffer wherever llama.cpp would
//    load the tensor there. That buffer interleaves the blocks of 4 or 8
//    rows, for kernels of its own, and ggml logs the layout it picks, like
//    q4_0_8x8, which this prints.
//
// ggml rounds the inputs to Q8_0 before it multiplies, so its products miss
// quantize's by about 0.5%. To check its kernels closely, this rounds the
// inputs as ggml does and multiplies NAME.decoded by them exactly: ggml's
// products must come within 1e-4 of that. quantize's must come within 1e-5
// of the exact product with the inputs as they are. Together, those leave
// ggml's products only as far from quantize's as rounding the inputs moves
// them. A code that a kernel reads wrong, like -128, which ggml's own
// quantizer never writes, moves the product by far more. Each error is
// relative: ‖got - expected‖ / ‖expected‖.
//
// Usage: check_ggml quantized_from_rust.gguf

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

static struct ggml_context * file_tensors;

// The tensor named `name` then `suffix`, which the file must hold.
static struct ggml_tensor * tensor(const char * name, const char * suffix) {
    char full_name[GGML_MAX_NAME];
    snprintf(full_name, sizeof full_name, "%s%s", name, suffix);
    struct ggml_tensor * found = ggml_get_tensor(file_tensors, full_name);
    if (!found) {
        fprintf(stderr, "the file has no tensor %s\n", full_name);
        exit(1);
    }
    return found;
}

// Each of `count` inputs dotted with each row of `weights`, added up in
// double precision.
static float * exact_product(const float * weights, int64_t rows, int64_t columns, const float * inputs,
                             int64_t count) {
    float * product = malloc(sizeof(float) * rows * count);
    for (int64_t vector = 0; vector < count; vector++) {
        for (int64_t row = 0; row < rows; row++) {
            double dot = 0;
            for (int64_t column = 0; column < columns; column++) {
                dot += (double) weights[row * columns + column] * inputs[vector * columns + column];
            }
            product[vector * rows + row] = (float) dot;
        }
    }
    return product;
}

// ‖got - expected‖ / ‖expected‖
static double relative_error(const float * got, const float * expected, int64_t count) {
    double difference = 0, size = 0;
    for (int64_t i = 0; i < count; i++) {
        difference += ((double) got[i] - expected[i]) * ((double) got[i] - expected[i]);
        size += (double) expected[i] * expected[i];
    }
    return sqrt(difference / size);
}

// `count` inputs rounded to Q8_0 by the quantizer that the CPU backend runs
// on inputs before it multiplies them, then decoded again.
static float * rounded_to_q8_0(const float * inputs, int64_t columns, int64_t count) {
    void * blocks = malloc(ggml_row_size(GGML_TYPE_Q8_0, columns) * count);
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(inputs, blocks, columns * count);
    float * rounded = malloc(sizeof(float) * columns * count);
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(blocks, rounded, columns * count);
    free(blocks);
    return rounded;
}

// The layout that the CPU_REPACK buffer last logged it repacked a tensor
// as, like "q4_0_8x8". ggml's other messages go to stderr, as they would.
static char repacked_as[32];

static void keep_repack_layout(enum ggml_log_level level, const char * text, void * user_data) {
    const char * with = strstr(text, "repack tensor") ? strstr(text, " with ") : NULL;
    if (with) {
        with += strlen(" with ");
        snprintf(repacked_as, sizeof repacked_as, "%.*s", (int) strcspn(with, "\n"), with);
    } else {
        fputs(text, stderr);
    }
    (void) level, (void) user_data;
}

// `weights` times the first `count` inputs on ggml's CPU backend, with the
// weights in a buffer of `buffer_type`. NULL where llama.cpp wouldn't put
// them there: it asks the backend whether it multiplies weights from that
// buffer by placing them in an empty one, as this does.
static float * ggml_product(ggml_backend_t backend, ggml_backend_buffer_type_t buffer_type,
                            const struct ggml_tensor * weights, const float * inputs, int64_t count) {
    struct ggml_init_params params = { 4 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
    struct ggml_context * weight_context = ggml_init(params);
    struct ggml_context * context = ggml_init(params);
    struct ggml_tensor * matrix = ggml_new_tensor_2d(weight_context, weights->type, weights->ne[0], weights->ne[1]);
    struct ggml_tensor * input = ggml_new_tensor_2d(context, GGML_TYPE_F32, weights->ne[0], count);
    struct ggml_tensor * product = ggml_mul_mat(context, matrix, input);

    matrix->buffer = ggml_backend_buft_alloc_buffer(buffer_type, 0);
    bool supported = ggml_backend_supports_op(backend, product);
    ggml_backend_buffer_free(matrix->buffer);
    matrix->buffer = NULL;
    float * out = NULL;
    if (supported) {
        ggml_backend_buffer_t weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weight_context, buffer_type);
        ggml_backend_tensor_set(matrix, weights->data, 0, ggml_nbytes(matrix));
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        ggml_backend_tensor_set(input, inputs, 0, ggml_nbytes(input));
        struct ggml_cgraph * graph = ggml_new_graph(context);
        ggml_build_forward_expand(graph, product);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "ggml couldn't compute the product\n");
            exit(1);
        }
        out = malloc(ggml_nbytes(product));
        ggml_backend_tensor_get(product, out, 0, ggml_nbytes(product));
        ggml_backend_buffer_free(buffer);
        ggml_backend_buffer_free(weight_buffer);
    }
    ggml_free(context);
    ggml_free(weight_context);
    return out;
}

// The CPU backend's CPU_REPACK buffer type, or NULL if it has none.
static ggml_backend_buffer_type_t repack_buffer_type(ggml_backend_dev_t cpu) {
    ggml_backend_dev_get_extra_bufts_t extra_buffer_types =
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu), "ggml_backend_dev_get_extra_bufts");
    for (ggml_backend_buffer_type_t * extra = extra_buffer_types(cpu); *extra; extra++) {
        if (strcmp(ggml_backend_buft_name(*extra), "CPU_REPACK") == 0) {
            return *extra;
        }
    }
    return NULL;
}

// Every check of Q4_0 or Q8_0 tensor `name`, `weights` in the file, printing
// what each finds. True if all of them pass.
static bool check(const char * name, const struct ggml_tensor * weights, const struct ggml_tensor * inputs,
                  ggml_backend_t backend, ggml_backend_buffer_type_t buffer_types[2]) {
    int64_t columns = weights->ne[0], rows = weights->ne[1];
    const float * decoded = tensor(name, ".decoded")->data;
    float * ggml_decoded = malloc(sizeof(float) * rows * columns);
    ggml_get_type_traits(weights->type)->to_float(weights->data, ggml_decoded, rows * columns);
    bool passed = memcmp(ggml_decoded, decoded, sizeof(float) * rows * columns) == 0;
    printf("\n%s, %lld × %lld: ggml decodes it to quantize's values, bit for bit: %s\n", name, (long long) rows,
           (long long) columns, passed ? "yes" : "NO");
    free(ggml_decoded);

    // quantize's products with the first input come first.
    const float * quantize_product = tensor(name, ".product")->data;
    const int64_t counts[] = { 1, inputs->ne[1] };
    for (int c = 0; c < 2; c++) {
        int64_t count = counts[c], outputs = rows * count;
        float * exact = exact_product(decoded, rows, columns, inputs->data, count);
        float * rounded_inputs = rounded_to_q8_0(inputs->data, columns, count);
        float * rounded = exact_product(decoded, rows, columns, rounded_inputs, count);
        double quantize_error = relative_error(quantize_product, exact, outputs);
        printf("  times %lld input%s: quantize's matmul is %.1e from the exact product, and rounding the inputs "
               "to Q8_0 moves that %.1e\n",
               (long long) count, count == 1 ? "" : "s", quantize_error, relative_error(rounded, exact, outputs));
        passed &= quantize_error < 1e-5;
        for (int b = 0; b < 2 && buffer_types[b]; b++) {
            repacked_as[0] = '\0';
            float * product = ggml_product(backend, buffer_types[b], weights, inputs->data, count);
            const char * buffer = ggml_backend_buft_name(buffer_types[b]);
            if (!product) {
                printf("    llama.cpp wouldn't load %s into the %s buffer on this CPU\n", name, buffer);
                continue;
            }
            double error = relative_error(product, rounded, outputs);
            printf("    ggml from the %s buffer", buffer);
            if (repacked_as[0]) {
                printf(", as %s,", repacked_as);
            }
            printf(" is %.1e from the rounded inputs' exact product, and %.1e from quantize's\n", error,
                   relative_error(product, quantize_product, outputs));
            passed &= error < 1e-4;
            free(product);
        }
        free(exact);
        free(rounded_inputs);
        free(rounded);
    }
    return passed;
}

int main(int argument_count, char ** arguments) {
    if (argument_count != 2) {
        fprintf(stderr, "usage: check_ggml quantized_from_rust.gguf\n");
        return 2;
    }
    ggml_log_set(keep_repack_layout, NULL);
    struct gguf_init_params load = { .no_alloc = false, .ctx = &file_tensors };
    struct gguf_context * file = gguf_init_from_file(arguments[1], load);
    if (!file) {
        fprintf(stderr, "ggml couldn't load %s\n", arguments[1]);
        return 1;
    }
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_type_t buffer_types[2] = {
        ggml_backend_cpu_buffer_type(),
        repack_buffer_type(ggml_backend_get_device(backend)),
    };
    printf("ggml loaded %s. CPU: AVX2 %d, AVX-512 %d, NEON %d, dot product %d, int8 matrix multiply %d\n",
           arguments[1], ggml_cpu_has_avx2(), ggml_cpu_has_avx512(), ggml_cpu_has_neon(), ggml_cpu_has_dotprod(),
           ggml_cpu_has_matmul_int8());

    const struct ggml_tensor * inputs = tensor("inputs", "");
    bool passed = true;
    for (int64_t t = 0; t < gguf_get_n_tensors(file); t++) {
        const char * name = gguf_get_tensor_name(file, t);
        const struct ggml_tensor * weights = ggml_get_tensor(file_tensors, name);
        if (weights->type == GGML_TYPE_Q4_0 || weights->type == GGML_TYPE_Q8_0) {
            passed &= check(name, weights, inputs, backend, buffer_types);
        }
    }
    ggml_backend_free(backend);
    gguf_free(file);
    ggml_free(file_tensors);
    printf("\n%s\n", passed ? "every check passed" : "SOME CHECK FAILED");
    return passed ? 0 : 1;
}
