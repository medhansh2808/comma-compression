/* Encoder companion to the inherited F26 RC64 decoder.
 * Include the decoder so constants and decoder entry points stay identical.
 */
#include "rc64_backend.c"

typedef struct {
    uint64_t low, high;
    size_t pending, bits, capacity;
    uint8_t *data;
    int finished, error;
} rc64_encoder;

static int enc_bit(rc64_encoder *e, unsigned bit) {
    size_t index = e->bits >> 3u;
    if (index >= e->capacity) {
        size_t capacity = e->capacity ? e->capacity * 2u : 4096u;
        uint8_t *p = (uint8_t *)realloc(e->data, capacity);
        if (!p) return e->error = -1;
        e->data = p;
        e->capacity = capacity;
    }
    if (!(e->bits & 7u)) e->data[index] = 0u;
    e->data[index] |= (uint8_t)(bit << (7u - (e->bits & 7u)));
    e->bits++;
    return 0;
}

static int enc_emit(rc64_encoder *e, unsigned bit) {
    if (enc_bit(e, bit)) return -1;
    while (e->pending) {
        if (enc_bit(e, 1u - bit)) return -1;
        e->pending--;
    }
    return 0;
}

void *rc64_encoder_create(void) {
    rc64_encoder *e = (rc64_encoder *)calloc(1u, sizeof(*e));
    if (e) e->high = RC64_TOP;
    return e;
}

void rc64_encoder_destroy(void *opaque) {
    rc64_encoder *e = (rc64_encoder *)opaque;
    if (e) { free(e->data); free(e); }
}

/* Deliberately follows rc64_decoder_decode_probabilities operation order.
 * Input probabilities are float32; truncation, winner ties and balancing
 * must agree exactly with the existing decoder.
 */
int rc64_encoder_encode_probabilities(
    void *opaque, const float *probabilities,
    size_t count, const int32_t *symbols
) {
    rc64_encoder *e = (rc64_encoder *)opaque;
    if (!e || e->finished || e->error ||
        (!probabilities && count) || (!symbols && count)) return -1;
    for (size_t index = 0; index < count; index++) {
        const float *row = probabilities + index * RC64_ALPHABET;
        uint32_t frequencies[RC64_ALPHABET];
        uint64_t sum = 0;
        double probability_sum = 0.0;
        unsigned winner = 0;
        int32_t target = symbols[index];
        if (target < 0 || target >= (int32_t)RC64_ALPHABET)
            return e->error = -2;
        for (unsigned k = 0; k < RC64_ALPHABET; k++) {
            double value = (double)row[k];
            if (!isfinite(value) || value <= 0.0 || value > 1.00002)
                return e->error = -3;
            probability_sum += value;
            if (row[k] > row[winner]) winner = k;
            uint64_t f = (uint64_t)(value * (double)RC64_TOTAL);
            if (f < 1u) f = 1u;
            frequencies[k] = (uint32_t)f;
            sum += f;
        }
        if (probability_sum < 0.99998 || probability_sum > 1.00002)
            return e->error = -4;
        int64_t adjusted = (int64_t)frequencies[winner] +
                           ((int64_t)RC64_TOTAL - (int64_t)sum);
        if (adjusted <= 0 || adjusted >= (int64_t)RC64_TOTAL)
            return e->error = -5;
        frequencies[winner] = (uint32_t)adjusted;
        uint64_t below = 0;
        for (unsigned k = 0; k < RC64_ALPHABET; k++) {
            if (!frequencies[k] || frequencies[k] >= RC64_TOTAL)
                return e->error = -5;
            if (k < (unsigned)target) below += frequencies[k];
        }
        uint64_t width = e->high - e->low + 1u;
        uint64_t lo = (uint64_t)(((__uint128_t)width * below) >> 31u);
        uint64_t hi = (uint64_t)(
            ((__uint128_t)width * (below + frequencies[target])) >> 31u);
        if (hi <= lo) return e->error = -6;
        e->high = e->low + hi - 1u;
        e->low += lo;
        for (;;) {
            if (e->high < RC64_HALF) {
                if (enc_emit(e, 0u)) return -7;
            } else if (e->low >= RC64_HALF) {
                if (enc_emit(e, 1u)) return -7;
                e->low -= RC64_HALF;
                e->high -= RC64_HALF;
            } else if (e->low >= RC64_FIRST_QTR &&
                       e->high < RC64_THIRD_QTR) {
                e->pending++;
                e->low -= RC64_FIRST_QTR;
                e->high -= RC64_FIRST_QTR;
            } else break;
            e->low <<= 1u;
            e->high = (e->high << 1u) | 1u;
        }
    }
    return 0;
}

int rc64_encoder_finish(void *opaque) {
    rc64_encoder *e = (rc64_encoder *)opaque;
    if (!e || e->error) return -1;
    if (!e->finished) {
        e->pending++;
        if (enc_emit(e, e->low < RC64_FIRST_QTR ? 0u : 1u)) return -2;
        e->finished = 1;
    }
    return 0;
}

const uint8_t *rc64_encoder_data(void *opaque) {
    rc64_encoder *e = (rc64_encoder *)opaque;
    return e ? e->data : NULL;
}

size_t rc64_encoder_size(void *opaque) {
    rc64_encoder *e = (rc64_encoder *)opaque;
    return e ? (e->bits + 7u) / 8u : 0u;
}
