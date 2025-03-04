/* Inference for Llama-2 Transformer model in pure C */

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#include <offload.h>
#include <immintrin.h>
#include <time.h>
#include "omp.h"

/* Decorations for offload data and functions prototypes */
#if defined( __INTEL_OFFLOAD) && ! defined (_WINHOST)
#define TARGET_MIC_ATTR __attribute__((target(mic)))
#define TARGET_MIC_PUSH _Pragma("offload_attribute(push,target(mic))")
#define TARGET_MIC_POP _Pragma("offload_attribute(pop)")

#elif defined(__INTEL_OFFLOAD) && defined (_WINHOST)
#define TARGET_MIC_ATTR __declspec(target(mic))
//for push use inline #pragma offload_attribute(push,target(mic))
//for pop use inline #pragma offload_attribute(pop)

#else
#define TARGET_MIC_ATTR
#define TARGET_MIC_PUSH
#define TARGET_MIC_POP
#endif

#ifndef _WINHOST
TARGET_MIC_PUSH
#else
#pragma offload_attribute(push,target(mic))
#endif
#include "mkl.h"
#ifndef _WINHOST
TARGET_MIC_POP
#else
#pragma offload_attribute(pop)
#endif

// use single precision for gemm
#define fptype_t float
#define xgemm sgemm
#define xgemv sgemv

#define ALIGNMENT 64
#define NUM_THREADS_MIC 244

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = (float*)_mm_malloc(p->dim * sizeof(float), ALIGNMENT);
    s->xb = (float*)_mm_malloc(p->dim * sizeof(float), ALIGNMENT);
    s->xb2 = (float*)_mm_malloc(p->dim * sizeof(float), ALIGNMENT);
    s->hb = (float*)_mm_malloc(p->hidden_dim * sizeof(float), ALIGNMENT);
    s->hb2 = (float*)_mm_malloc(p->hidden_dim * sizeof(float), ALIGNMENT);
    s->q = (float*)_mm_malloc(p->dim * sizeof(float), ALIGNMENT);
    s->key_cache = (float*)_mm_malloc(p->n_layers * p->seq_len * kv_dim * sizeof(float), ALIGNMENT);
    s->value_cache = (float*)_mm_malloc(p->n_layers * p->seq_len * kv_dim * sizeof(float), ALIGNMENT);
    s->att = (float*)_mm_malloc(p->n_heads * p->seq_len * sizeof(float), ALIGNMENT);
    s->logits = (float*)_mm_malloc(p->vocab_size * sizeof(float), ALIGNMENT);

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    _mm_free(s->x);
    _mm_free(s->xb);
    _mm_free(s->xb2);
    _mm_free(s->hb);
    _mm_free(s->hb2);
    _mm_free(s->q);
    _mm_free(s->att);
    _mm_free(s->logits);
    _mm_free(s->key_cache);
    _mm_free(s->value_cache);

}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

TARGET_MIC_ATTR int ONE = 1;
TARGET_MIC_ATTR float rLN2 = 1.44269504088896340735992468100189213742664595415298593413544940693110921918f;

#ifndef __MIC__

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    // Calculate sum of squares with vectorization
    /*
    #pragma omp parallel for simd reduction(+:ss)
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    //#pragma omp barrier
    */
    ss = sdot(&size, x, &ONE, x, &ONE);

    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    
    // Normalize and scale with vectorization
    #pragma omp parallel for simd
    //#pragma omp simd
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
    //#pragma omp barrier
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i++) {
		x[i] = expf(x[i] - max_val);
        //x[i] = exp2f(x[i] - max_val) * rLN2;
        sum += x[i];
    }
    //#pragma omp barrier

    // normalize
    #pragma omp parallel for simd
    //#pragma omp simd
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
    // #pragma omp barrier
}

#else

TARGET_ATTRIBUTE // MIC attribute
void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;

    // Calculate sum of squares with vectorization
    /*
    #pragma omp parallel for simd reduction(+:ss)
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    //#pragma omp barrier
    */
    ss = sdot(&size, x, &ONE, x, &ONE);

    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    
    // Normalize and scale with vectorization
    /*
    #pragma omp parallel for simd
    //#pragma omp simd
    for (int j = 0; j < size; j+=4) {
        o[j] = weight[j] * (ss * x[j]);
        o[j + 1] = weight[j + 1] * (ss * x[j + 1]);
        o[j + 2] = weight[j + 2] * (ss * x[j + 2]);
        o[j + 3] = weight[j + 3] * (ss * x[j + 3]);
    }
    //#pragma omp barrier
    */

    __m512 ss_load = _mm512_set1_ps(ss); 
    int size_aligned = size / 16 * 16; 
    #pragma omp parallel for
    //#pragma omp simd
    for (int j = 0; j < size_aligned; j+=16) {
        __m512 weight_load = _mm512_load_ps(weight + j); 
        __m512 x_load = _mm512_load_ps(x + j); 

        x_load = _mm512_mul_ps(ss_load, x_load);
        x_load = _mm512_mul_ps(weight_load, x_load);
        _mm512_store_ps(o + j, x_load);  
    }
    //#pragma omp barrier
    // Remains
    for (int j = size_aligned; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

TARGET_ATTRIBUTE // MIC attribute
void softmax(float* x, int size) {
    // find max value (for numerical stability)

    __m512 max_val_load = _mm512_load_ps(x); 
    __m512 x_load;
    int size_aligned = size / 16 * 16; 
    for (int i = 16; i < size_aligned; i+=16) {
        x_load = _mm512_load_ps(x + i);
        // max_val_load[j] = x_load[j] > max_val_load[j] ? x_load[j] : max_val_load[j]; 
        max_val_load = _mm512_max_ps(max_val_load, x_load); 
    }
    float max_val = _mm512_reduce_max_ps(max_val_load); 
    // remains 
    for (int i = size_aligned; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i]; 
        }
    }

    // exp and sum
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i++) {
		x[i] = expf(x[i] - max_val);
        //x[i] = exp2f(x[i] - max_val) * rLN2;
        sum += x[i];
    }
    //#pragma omp barrier

    // It's not good to do intrinsic here. And unroll with step = 4 is good enough. 
    // normalize
    #pragma omp parallel for simd 
    //#pragma omp simd
    for (int i = 0; i < size; i+=4) {
        x[i] /= sum;
        x[i + 1] /= sum;
        x[i + 2] /= sum;
        x[i + 3] /= sum;
    }
    //#pragma omp barrier
}

#endif

// AVX2 version. Somehow this is faster than the MKL version.
// Credit: https://github.com/trholding/llama2.c/blob/5b2822e1896c72486f38214e8a3cf5fb7c7a84e9/run.c#L413
void matmul_avx2(float* xout, const float* x, const float* w, int n, int d) {
    int nn = n / 32 * 32;  // ensure n is a multiple of 8
    int i;
    __m256 sum_vec;
    #pragma omp parallel for private(i, sum_vec)
    for (i = 0; i < d; i++) {
        sum_vec = _mm256_setzero_ps(); // for AVX2, sum of 8 floats
        int i_n = i * n;
        #pragma omp simd
        for (int j = 0; j < nn; j += 32) {
            // Load 32 values from w and x
            __m256 w_vec0 = _mm256_loadu_ps(&w[i_n + j]);
            __m256 w_vec1 = _mm256_loadu_ps(&w[i_n + j + 8]);
            __m256 w_vec2 = _mm256_loadu_ps(&w[i_n + j + 16]);
            __m256 w_vec3 = _mm256_loadu_ps(&w[i_n + j + 24]);
            __m256 x_vec0 = _mm256_loadu_ps(&x[j]);
            __m256 x_vec1 = _mm256_loadu_ps(&x[j + 8]);
            __m256 x_vec2 = _mm256_loadu_ps(&x[j + 16]);
            __m256 x_vec3 = _mm256_loadu_ps(&x[j + 24]);

            // Multiply and accumulate
            __m256 prod_vec0 = _mm256_mul_ps(w_vec0, x_vec0);
            __m256 prod_vec1 = _mm256_mul_ps(w_vec1, x_vec1);
            __m256 prod_vec2 = _mm256_mul_ps(w_vec2, x_vec2);
            __m256 prod_vec3 = _mm256_mul_ps(w_vec3, x_vec3);
            sum_vec = _mm256_add_ps(sum_vec, prod_vec0);
            sum_vec = _mm256_add_ps(sum_vec, prod_vec1);
            sum_vec = _mm256_add_ps(sum_vec, prod_vec2);
            sum_vec = _mm256_add_ps(sum_vec, prod_vec3);
        }

        // Perform horizontal add
        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        float vals[8];
        _mm256_storeu_ps(vals, sum_vec);
        float val = vals[0] + vals[4];

        // handle remainder if n is not a multiple of 8
        int j;
        #pragma omp simd reduction(+:val)
        for (j = nn; j < n; j++) {
            val += w[i_n + j] * x[j];
        }
        xout[i] = val;
    }
}

//IMCI implementation

TARGET_ATTRIBUTE // MIC attribute
void matmul_imci(float* xout, const float* x, const float* w, int n, int d) {
    int nn = n / 64 * 64;  // ensure n is a multiple of 16
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        __m512 sum_vec = _mm512_setzero_ps(); // for IMCI, sum of 16 floats
        int i_n = i * n;
        #pragma omp simd
        for (int j = 0; j < nn; j += 64) {
            // Load 64 values from w and x
            __m512 w_vec0 = _mm512_load_ps(&w[i_n + j]);
            __m512 w_vec1 = _mm512_load_ps(&w[i_n + j + 16]);
            __m512 w_vec2 = _mm512_load_ps(&w[i_n + j + 32]);
            __m512 w_vec3 = _mm512_load_ps(&w[i_n + j + 48]);
            __m512 x_vec0 = _mm512_load_ps(&x[j]);
            __m512 x_vec1 = _mm512_load_ps(&x[j + 16]);
            __m512 x_vec2 = _mm512_load_ps(&x[j + 32]);
            __m512 x_vec3 = _mm512_load_ps(&x[j + 48]);

            // Multiply and accumulate
            __m512 prod_vec0 = _mm512_mul_ps(w_vec0, x_vec0);
            __m512 prod_vec1 = _mm512_mul_ps(w_vec1, x_vec1);
            __m512 prod_vec2 = _mm512_mul_ps(w_vec2, x_vec2);
            __m512 prod_vec3 = _mm512_mul_ps(w_vec3, x_vec3);
            sum_vec = _mm512_add_ps(sum_vec, prod_vec0);
            sum_vec = _mm512_add_ps(sum_vec, prod_vec1);
            sum_vec = _mm512_add_ps(sum_vec, prod_vec2);
            sum_vec = _mm512_add_ps(sum_vec, prod_vec3);
        }

        // Reduce add to get result
        float val = _mm512_reduce_add_ps(sum_vec);

        // handle remainder if n is not a multiple of 16
        int j;
        #pragma omp simd reduction(+:val)
        for (j = nn; j < n; j++) {
            val += w[i_n + j] * x[j];
        }
        xout[i] = val;
    }
}

// Naive implementation

void matmul_naive(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        #pragma omp simd
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
    //#pragma omp barrier
}


TARGET_ATTRIBUTE // MIC attribute
void matmul_naive_mic(float* xout, float* x, float* w, int n, int d) {
	// W (d,n) @ x (n,) -> xout (d,)
	// by far the most amount of time is spent inside this little function
	#pragma omp parallel for simd
	for (int i = 0; i < d; i++) {
		float val = 0.0f;
		#pragma omp simd
		for (int j = 0; j < n; j++) {
			val += w[i * n + j] * x[j];
		}
		xout[i] = val;
	}
	//#pragma omp barrier
}

TARGET_MIC_ATTR float MATMUL_ALPHA = 1.0; 
TARGET_MIC_ATTR float MATMUL_BETA = 0.0; 
TARGET_MIC_ATTR int MATMUL_ONE = 1; 
TARGET_MIC_ATTR char MATMUL_TRANS = 'N';

// MKL version
void matmul_mkl(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // Wrong Answers. 
    //sgemv(&MATMUL_TRANS, &d, &n, &MATMUL_ALPHA, w, &d, x, &MATMUL_ONE, &MATMUL_BETA, xout, &MATMUL_ONE);
    //xgemm(&MATMUL_TRANS, &MATMUL_TRANS, &d, &MATMUL_ONE, &n, &MATMUL_ALPHA, w, &d, x, &n, &MATMUL_BETA, xout, &d);
    // Answers Correct. 
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d, n, 1.0f, w, n, x, 1, 0.0f, xout, 1);
}

TARGET_ATTRIBUTE // MIC attribute
void matmul_mkl_mic(float* xout, float* x, float* w, int n, int d) {
	// W (d,n) @ x (n,) -> xout (d,)
    // Wrong Answers. 
    //sgemv(&MATMUL_TRANS, &d, &n, &MATMUL_ALPHA, w, &d, x, &MATMUL_ONE, &MATMUL_BETA, xout, &MATMUL_ONE);
    //sgemm(&MATMUL_TRANS, &MATMUL_TRANS, &d, &MATMUL_ONE, &n, &MATMUL_ALPHA, w, &d, x, &n, &MATMUL_BETA, xout, &d);
    // Answers Correct. 
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d, n, 1.0f, w, n, x, 1, 0.0f, xout, 1);
}

TARGET_ATTRIBUTE // MIC attribute
void forward_mic(
    float *x, 
    int dim,
    int kv_dim,
    int kv_mul,
    int hidden_dim,
    int head_size,
    int p_n_layers, 
    int p_seq_len,
    int p_n_kv_heads,
    int p_n_heads,
    int p_dim, 
    int p_vocab_size,
    float *w_rms_att_weight,
    float *w_wq,
    float *w_wk,
    float *w_wv,
    float *w_wo, 
    float *w_rms_ffn_weight,
    float *w_w1,
    float *w_w2,
    float *w_w3,
    float *w_rms_final_weight,
    float *w_wcls, 
    float *s_q,
    float *s_key_cache,
    float *s_value_cache,
    float *s_xb,
    float *s_xb2,
    float *s_hb,
    float *s_hb2,
    float *s_att, 
    float *s_logits,
    int token, 
    int pos, 
    int offloaded_begin,
    int offloaded_end, 
    int matmul(float*, float*, float*, int, int)

) {
    //omp_set_num_threads(NUM_THREADS_MIC);
    for(unsigned long long l = offloaded_begin; l < offloaded_end; l++) {
        // attention rmsnorm
        rmsnorm(s_xb, x, w_rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int loff = l * p_seq_len * kv_dim; // kv cache layer offset for convenience
        float *s_k = s_key_cache + loff + pos * kv_dim;
        float *s_v = s_value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s_q, s_xb, w_wq + l*dim*dim, dim, dim);
		// Calculate new KV$, located at s_{key,value}_cache + loff + pos * kv_dim, length = kv_dim(fp32)
        matmul(s_k, s_xb, w_wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s_v, s_xb, w_wv + l*dim*kv_dim, dim, kv_dim);


        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        #pragma omp parallel for simd
        for (int i = 0; i < kv_dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);

            s_q[i]   = s_q[i] * fcr - s_q[i+1] * fci;
            s_q[i+1] = s_q[i] * fci + s_q[i+1] * fcr;
            s_k[i]   = s_k[i] * fcr - s_k[i+1] * fci;
            s_k[i+1] = s_k[i] * fci + s_k[i+1] * fcr;
        }
        //#pragma omp barrier

        #pragma omp parallel for simd
        for (int i = kv_dim; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);

            s_q[i]   = s_q[i] * fcr - s_q[i+1] * fci;
            s_q[i+1] = s_q[i] * fci + s_q[i+1] * fcr;
        }
        //#pragma omp barrier

        // multihead attention. iterate over all heads
        // Should we parallel at here? Threads on MIC is way more than heads 
        // In practise, for small models, here we should. For large models, here we shouldn't. 
        #pragma omp parallel for 
        for (int h = 0; h < p_n_heads; h++) {
            // get the query vector for this head
            float* q = s_q + h * head_size;
            // attention scores for this head
            float* att = s_att + h * p_seq_len;
            // iterate over all timesteps, including the current one
            //#pragma omp parallel for
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s_key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k

                // Don't use MKL here, head_size is 64
				//score = sdot(&head_size, q, &ONE, k, &ONE);
                
                /*
                float score = 0.0f;
                #pragma omp simd
                for (int i = 0; i < head_size; i+=4) {
                    score += q[i] * k[i];
                    score += q[i + 1] * k[i + 1];
                    score += q[i + 2] * k[i + 2];
                    score += q[i + 3] * k[i + 3];
                }
                */
                
                // Intrinsic
                int head_size_aligned = head_size / 16 * 16; 
                __m512 score_load = _mm512_setzero_ps(); 
                #pragma omp simd
                for (int i = 0; i < head_size_aligned; i+=16) {
                    __m512 q_load = _mm512_load_ps(q + i); 
                    __m512 k_load = _mm512_load_ps(k + i); 
                    __m512 qk_product = _mm512_mul_ps(q_load, k_load); 
                    score_load = _mm512_add_ps(score_load, qk_product); 
                }
                float score = _mm512_reduce_add_ps(score_load); 
                // remains
                for (int i = head_size_aligned; i < head_size; i++) {
                    score += q[i] * k[i];
                }

                // save the score to the attention buffer
                att[t] = score / sqrtf(head_size);
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s_xb + h * head_size;
            //memset(xb, 0, head_size * sizeof(float));
            // t = 0
            float *v = s_value_cache + loff + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = att[0];
            // accumulate the weighted value into xb
            #pragma omp simd
            for (int i = 0; i < head_size; i+=4) {
                xb[i] = a * v[i];
                xb[i + 1] = a * v[i + 1];
                xb[i + 2] = a * v[i + 2];
                xb[i + 3] = a * v[i + 3];
            }
            // t = 1..pos
            for (int t = 1; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                v = s_value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                a = att[t];
                // accumulate the weighted value into xb
                #pragma omp simd
                for (int i = 0; i < head_size; i+=4) {
                    xb[i] += a * v[i];
                    xb[i + 1] += a * v[i + 1];
                    xb[i + 2] += a * v[i + 2];
                    xb[i + 3] += a * v[i + 3];
                }
            }
        }
        //#pragma omp barrier

        // final matmul to get the output of the attention
        matmul(s_xb2, s_xb, w_wo + l*dim*dim, dim, dim);


        // residual connection back into x
        // Intrinsic
        #pragma omp parallel for simd 
        for (int i = 0; i < dim; i+=16) {
            // x[i] += s_xb2[i];
            __m512 x_load = _mm512_load_ps(x + i); 
            __m512 s_xb2_load = _mm512_load_ps(s_xb2 + i); 
            x_load = _mm512_add_ps(x_load, s_xb2_load); 
            _mm512_store_ps(x + i, x_load); 
        }

        // ffn rmsnorm
        rmsnorm(s_xb, x, w_rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s_hb, s_xb, w_w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s_hb2, s_xb, w_w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        #pragma omp parallel for simd
        for (int i = 0; i < hidden_dim; i++) {
            float val = s_hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s_hb2[i];
            s_hb[i] = val;
        }
        //#pragma omp barrier

        // final matmul to get the output of the ffn
        matmul(s_xb, s_hb, w_w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        // Intrinsic
        #pragma omp parallel for simd 
        for (int i = 0; i < dim; i+=16) {
            // x[i] += s_xb[i];
            __m512 x_load = _mm512_load_ps(x + i); 
            __m512 s_xb_load = _mm512_load_ps(s_xb + i); 
            x_load = _mm512_add_ps(x_load, s_xb_load); 
            _mm512_store_ps(x + i, x_load); 
        }
    }
    if (offloaded_end == p_n_layers) {
        rmsnorm(x, x, w_rms_final_weight, dim);
        // get the logits
        matmul(s_logits, x, w_wcls, dim, p_vocab_size);
    }
}


void forward_cpu(
    Transformer* transformer, 
    int token, 
    int pos, 
    int offloaded_begin,
    int offloaded_end, 
    void *matmul(float*, float*, float*, int, int)
) {
    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    // copy the token embedding into x
 
    // CPU part
    for(unsigned long long l = offloaded_begin; l < offloaded_end; l++) {
        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        #pragma omp parallel for simd
        for (int i = 0; i < kv_dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);

            s->q[i]   = s->q[i] * fcr - s->q[i+1] * fci;
            s->q[i+1] = s->q[i] * fci + s->q[i+1] * fcr;
            s->k[i]   = s->k[i] * fcr - s->k[i+1] * fci;
            s->k[i+1] = s->k[i] * fci + s->k[i+1] * fcr;
        }
        // Rotate q only. 
        #pragma omp parallel for simd
        for (int i = kv_dim; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);

            s->q[i]   = s->q[i] * fcr - s->q[i+1] * fci;
            s->q[i+1] = s->q[i] * fci + s->q[i+1] * fcr;
        }
        //#pragma omp barrier

        // multihead attention. iterate over all heads
        #pragma omp parallel for 
        for (int h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            //#pragma omp parallel for
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
				score = sdot(&head_size, q, &ONE, k, &ONE);
                /*
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                */
                // save the score to the attention buffer
                att[t] = score / sqrtf(head_size);
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            //memset(xb, 0, head_size * sizeof(float));
            // t = 0
            float *v = s->value_cache + loff + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = att[0];
            // accumulate the weighted value into xb
            #pragma omp simd
            for (int i = 0; i < head_size; i++) {
                xb[i] = a * v[i];
            }
            // t = 1..pos
			for (int t = 1; t <= pos; t++) {
				// get the value vector for this head and at this timestep
				v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
				// get the attention weight for this timestep
				a = att[t];
				// accumulate the weighted value into xb
				#pragma omp simd
				for (int i = 0; i < head_size; i++) {
					xb[i] += a * v[i];
				}
			}
        }
        //#pragma omp barrier
        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        #pragma omp parallel for simd
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }
        //#pragma omp barrier

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        #pragma omp parallel for simd
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        //#pragma omp barrier

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        #pragma omp parallel for simd
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
        //#pragma omp barrier
    }
    
    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { _mm_free(t->vocab[i]); }
    _mm_free(t->vocab);
    _mm_free(t->vocab_scores);
    _mm_free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are 
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps, int offloaded_layers, 
                void *matmul_cpu(float*, float*, float*, int, int), int matmul_selecting_mic) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence

	Config* p = &transformer->config;
	TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
	// a few convenience variables
	int dim = p->dim;
	int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
	int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
	int hidden_dim = p->hidden_dim;
	int head_size = dim / p->n_heads;
	int p_n_layers = p->n_layers;
	int p_seq_len = p->seq_len;
	int p_n_kv_heads = p->n_kv_heads;
	int p_n_heads = p->n_heads;
	int p_dim = p->dim;
	int p_vocab_size = p->vocab_size;
	float *w_rms_att_weight = w->rms_att_weight;
	float *w_wq = w->wq;
	float *w_wk = w->wk;
	float *w_wv = w->wv;
	float *w_wo = w->wo;
	float *w_rms_ffn_weight = w->rms_ffn_weight;
	float *w_w1 = w->w1;
	float *w_w2 = w->w2;
	float *w_w3 = w->w3;
	float *w_rms_final_weight = w->rms_final_weight;
	float *w_wcls = w->wcls;
	float *s_q = s->q;
	float *s_key_cache = s->key_cache;
	float *s_value_cache = s->value_cache;
	float *s_xb = s->xb;
	float *s_xb2 = s->xb2;
	float *s_hb = s->hb;
	float *s_hb2 = s->hb2;
	float *s_att = s->att;
    float *s_logits = s->logits;

    // Allocate memory on MIC
    if (offloaded_layers > 0) {
        char *sign;

        printf("Mallocing on MIC\n");
        #pragma offload target(mic : 0) signal(sign) \
            nocopy(x : length(dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_rms_att_weight : length(offloaded_layers * dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_wq : length(offloaded_layers*dim*dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_wk : length(offloaded_layers*dim*kv_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_wv : length(offloaded_layers*dim*kv_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_wo : length(offloaded_layers*dim*dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_rms_ffn_weight : length(offloaded_layers*dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_w1 : length(offloaded_layers*dim*hidden_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_w2 : length(offloaded_layers*dim*hidden_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_w3 : length(offloaded_layers*dim*hidden_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_rms_final_weight : length(dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_wcls : length(dim*p_vocab_size) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_q : length(dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_key_cache : length(offloaded_layers * p_seq_len * kv_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_value_cache : length(offloaded_layers * p_seq_len * kv_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_xb : length(dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_xb2 : length(dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_hb : length(hidden_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_hb2 : length(hidden_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_att : length(p_n_heads * p_seq_len) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_logits : length(p_vocab_size) alloc_if(1) free_if(0) align(ALIGNMENT))
        {}
        #pragma offload_wait target(mic : 0) wait(sign)
        printf("Mallocing on MIC done\n");

    }

    while (pos < steps) {
        // copy the token embedding into x
        float* content_row = w->token_embedding_table + token * dim;
        memcpy(x, content_row, dim*sizeof(float));

        if (offloaded_layers > 0) {
            char *sign;

            #pragma offload target(mic : 0) signal(sign) \
                in(dim) \
                in(kv_dim) \
                in(kv_mul) \
                in(hidden_dim) \
                in(head_size) \
                in(p_n_layers) \
                in(offloaded_layers) \
                in(p_seq_len) \
                in(p_n_heads) \
                in(p_dim) \
                in(p_vocab_size) \
                inout(x : length(dim) alloc_if(0) free_if(0) align(ALIGNMENT)) \
                nocopy(w_rms_att_weight : alloc_if(0) free_if(0)) \
                nocopy(w_wq : alloc_if(0) free_if(0)) \
                nocopy(w_wk : alloc_if(0) free_if(0)) \
                nocopy(w_wv : alloc_if(0) free_if(0)) \
                nocopy(w_wo : alloc_if(0) free_if(0)) \
                nocopy(w_rms_ffn_weight : alloc_if(0) free_if(0)) \
                nocopy(w_w1 : alloc_if(0) free_if(0)) \
                nocopy(w_w2 : alloc_if(0) free_if(0)) \
                nocopy(w_w3 : alloc_if(0) free_if(0)) \
                nocopy(w_rms_final_weight : alloc_if(0) free_if(0)) \
                nocopy(w_wcls : alloc_if(0) free_if(0)) \
                nocopy(s_q : alloc_if(0) free_if(0)) \
                nocopy(s_key_cache : length(offloaded_layers * p_seq_len * kv_dim) alloc_if(0) free_if(0)) \
                nocopy(s_value_cache : length(offloaded_layers * p_seq_len * kv_dim) alloc_if(0) free_if(0)) \
                nocopy(s_xb : alloc_if(0) free_if(0)) \
                nocopy(s_xb2 : alloc_if(0) free_if(0)) \
                nocopy(s_hb : alloc_if(0) free_if(0)) \
                nocopy(s_hb2 : alloc_if(0) free_if(0)) \
                nocopy(s_att : alloc_if(0) free_if(0)) \
                nocopy(s_logits : alloc_if(0) free_if(0)) \
                in(matmul_selecting_mic)
            {
                void *matmul_mic;
                if (matmul_selecting_mic == 0) {
                    matmul_mic = matmul_naive_mic;
                } else if (matmul_selecting_mic == 1) {
                    matmul_mic = matmul_mkl_mic;
                } else if (matmul_selecting_mic == 2) {
                    matmul_mic = matmul_imci; 
                }

                forward_mic(
                    x, 
                    dim,
                    kv_dim,
                    kv_mul,
                    hidden_dim,
                    head_size,
                    p_n_layers, 
                    p_seq_len,
                    p_n_kv_heads,
                    p_n_heads,
                    p_dim, 
                    p_vocab_size,
                    w_rms_att_weight,
                    w_wq,
                    w_wk,
                    w_wv,
                    w_wo, 
                    w_rms_ffn_weight,
                    w_w1,
                    w_w2,
                    w_w3,
                    w_rms_final_weight,
                    w_wcls, 
                    s_q,
                    s_key_cache,
                    s_value_cache,
                    s_xb,
                    s_xb2,
                    s_hb,
                    s_hb2,
                    s_att, 
                    s_logits,
                    token, 
                    pos, 
                    0,
                    offloaded_layers,
                    matmul_mic
                );
            }
            #pragma offload_wait target(mic : 0) wait(sign)

            if (offloaded_layers < p_n_layers) {
                int *sign;
                // only transmit new KV$. 
				for (int l = 0; l < offloaded_layers; l++) {
                    // key and value point to the kv cache
                    int loff = l * p_seq_len * kv_dim; // kv cache layer offset for convenience
                    float *s_k = s_key_cache + loff + pos * kv_dim;
                    float *s_v = s_value_cache + loff + pos * kv_dim;

                    #pragma offload target(mic : 0) signal(sign) \
                        out(s_k : length(kv_dim) alloc_if(0) free_if(0)) \
                        out(s_v : length(kv_dim) alloc_if(0) free_if(0)) 
                    {}
                    //#pragma offload_wait target(mic : 0) wait(sign)
				}
                
                // only transmit new attentions
                for (int h = 0; h < p_n_heads; h++) {
                    float *s_att_h = s_att + h * p_seq_len; 
                    #pragma offload target(mic : 0) signal(sign) \
                        out(s_att_h : length(pos) alloc_if(0) free_if(0))
                    {}
                    #pragma offload_wait target(mic : 0) wait(sign)
                }
            }
        }

        // forward the transformer to get logits for the next token
        if (offloaded_layers < p_n_layers) {
            forward_cpu(transformer, token, pos, offloaded_layers, p->n_layers, matmul_cpu);
        } else {
            int *sign;
            #pragma offload target(mic : 0) signal(sign) \
                out(s_logits : length(p_vocab_size) alloc_if(0) free_if(0))
            {}
            //#pragma offload_wait target(mic : 0) wait(sign)
        }
        float* logits = transformer->state.logits;

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        printf("%s", piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // Free the allocated memory on MIC
    if (offloaded_layers > 0) {
        char *sign;

        printf("Freeing on MIC\n");
        #pragma offload target(mic : 0) signal(sign) \
            nocopy(x : alloc_if(0) free_if(1)) \
            nocopy(w_rms_att_weight :alloc_if(0) free_if(1)) \
            nocopy(w_wq : alloc_if(0) free_if(1)) \
            nocopy(w_wk : alloc_if(0) free_if(1)) \
            nocopy(w_wv : alloc_if(0) free_if(1)) \
            nocopy(w_wo : alloc_if(0) free_if(1)) \
            nocopy(w_rms_ffn_weight : alloc_if(0) free_if(1)) \
            nocopy(w_w1 : alloc_if(0) free_if(1)) \
            nocopy(w_w2 : alloc_if(0) free_if(1)) \
            nocopy(w_w3 : alloc_if(0) free_if(1)) \
            nocopy(w_rms_final_weight : alloc_if(0) free_if(1)) \
            nocopy(w_wcls : alloc_if(0) free_if(1)) \
            nocopy(s_q : alloc_if(0) free_if(1)) \
            nocopy(s_key_cache : alloc_if(0) free_if(1)) \
            nocopy(s_value_cache : alloc_if(0) free_if(1)) \
            nocopy(s_xb : alloc_if(0) free_if(1)) \
            nocopy(s_xb2 : alloc_if(0) free_if(1)) \
            nocopy(s_hb : alloc_if(0) free_if(1)) \
            nocopy(s_hb2 : alloc_if(0) free_if(1)) \
            nocopy(s_att : alloc_if(0) free_if(1)) \
            nocopy(s_logits : alloc_if(0) free_if(1))
        {}
        #pragma offload_wait target(mic : 0) wait(sign)
        printf("Freeing on MIC done\n");
    }

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "%d tokens generated in %.3f s, achieved tok/s: %f\n", (pos-1), (double)(end-start) / 1000, (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps, int offloaded_layers, 
          void *matmul_cpu(float*, float*, float*, int, int), 
          int matmul_selecting_mic) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    long start_time = 0;  // used to time our code, only initialized after first iteration
    long end_time = 0; 
    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    int pos_previous = 0;

	Config* p = &transformer->config;
	TransformerWeights* w = &transformer->weights;
	RunState* s = &transformer->state;
	float *x = s->x;
	// a few convenience variables
	int dim = p->dim;
	int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
	int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
	int hidden_dim = p->hidden_dim;
	int head_size = dim / p->n_heads;
	int p_n_layers = p->n_layers;
	int p_seq_len = p->seq_len;
	int p_n_kv_heads = p->n_kv_heads;
	int p_n_heads = p->n_heads;
	int p_dim = p->dim;
	int p_vocab_size = p->vocab_size;
	float *w_rms_att_weight = w->rms_att_weight;
	float *w_wq = w->wq;
	float *w_wk = w->wk;
	float *w_wv = w->wv;
	float *w_wo = w->wo;
	float *w_rms_ffn_weight = w->rms_ffn_weight;
	float *w_w1 = w->w1;
	float *w_w2 = w->w2;
	float *w_w3 = w->w3;
	float *w_rms_final_weight = w->rms_final_weight;
	float *w_wcls = w->wcls;
	float *s_q = s->q;
	float *s_key_cache = s->key_cache;
	float *s_value_cache = s->value_cache;
	float *s_xb = s->xb;
	float *s_xb2 = s->xb2;
	float *s_hb = s->hb;
	float *s_hb2 = s->hb2;
	float *s_att = s->att;
    float *s_logits = s->logits;

    // Allocate memory on MIC
	if (offloaded_layers > 0) {
        char *sign;

        printf("Mallocing on MIC\n");
        #pragma offload target(mic : 0) signal(sign) \
            nocopy(x : length(dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_rms_att_weight : length(offloaded_layers * dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_wq : length(offloaded_layers*dim*dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_wk : length(offloaded_layers*dim*kv_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_wv : length(offloaded_layers*dim*kv_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_wo : length(offloaded_layers*dim*dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_rms_ffn_weight : length(offloaded_layers*dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_w1 : length(offloaded_layers*dim*hidden_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_w2 : length(offloaded_layers*dim*hidden_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_w3 : length(offloaded_layers*dim*hidden_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_rms_final_weight : length(dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            in(w_wcls : length(dim*p_vocab_size) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_q : length(dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_key_cache : length(offloaded_layers * p_seq_len * kv_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_value_cache : length(offloaded_layers * p_seq_len * kv_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_xb : length(dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_xb2 : length(dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_hb : length(hidden_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_hb2 : length(hidden_dim) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_att : length(p_n_heads * p_seq_len) alloc_if(1) free_if(0) align(ALIGNMENT)) \
            nocopy(s_logits : length(p_vocab_size) alloc_if(1) free_if(0) align(ALIGNMENT))
        {}
        #pragma offload_wait target(mic : 0) wait(sign)
        printf("Mallocing on MIC done\n");
    }

    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                if (pos) {
                    end_time = time_in_ms();
                    fprintf(stderr, "\n%d tokens generated in %.3f s, achieved tok/s: %f\n", 
                                        (pos - pos_previous), 
                                        (double)(end_time -start_time) / 1000, 
                                        (pos - pos_previous) / (double)(end_time - start_time) *1000);
                }
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
                start_time = time_in_ms();
                pos_previous = pos; 
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);

            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // copy the token embedding into x
        float* content_row = w->token_embedding_table + token * dim;
        memcpy(x, content_row, dim*sizeof(float));

        if (offloaded_layers > 0) {
            char *sign;


            #pragma offload target(mic : 0) signal(sign) \
                in(dim) \
                in(kv_dim) \
                in(kv_mul) \
                in(hidden_dim) \
                in(head_size) \
                in(p_n_layers) \
                in(offloaded_layers) \
                in(p_seq_len) \
                in(p_n_heads) \
                in(p_dim) \
                in(p_vocab_size) \
                inout(x : length(dim) alloc_if(0) free_if(0) align(ALIGNMENT)) \
                nocopy(w_rms_att_weight : alloc_if(0) free_if(0)) \
                nocopy(w_wq : alloc_if(0) free_if(0)) \
                nocopy(w_wk : alloc_if(0) free_if(0)) \
                nocopy(w_wv : alloc_if(0) free_if(0)) \
                nocopy(w_wo : alloc_if(0) free_if(0)) \
                nocopy(w_rms_ffn_weight : alloc_if(0) free_if(0)) \
                nocopy(w_w1 : alloc_if(0) free_if(0)) \
                nocopy(w_w2 : alloc_if(0) free_if(0)) \
                nocopy(w_w3 : alloc_if(0) free_if(0)) \
                nocopy(w_rms_final_weight : alloc_if(0) free_if(0)) \
                nocopy(w_wcls : alloc_if(0) free_if(0)) \
                nocopy(s_q : alloc_if(0) free_if(0)) \
                nocopy(s_key_cache : length(offloaded_layers * p_seq_len * kv_dim) alloc_if(0) free_if(0)) \
                nocopy(s_value_cache : length(offloaded_layers * p_seq_len * kv_dim) alloc_if(0) free_if(0)) \
                nocopy(s_xb : alloc_if(0) free_if(0)) \
                nocopy(s_xb2 : alloc_if(0) free_if(0)) \
                nocopy(s_hb : alloc_if(0) free_if(0)) \
                nocopy(s_hb2 : alloc_if(0) free_if(0)) \
                nocopy(s_att : alloc_if(0) free_if(0)) \
                nocopy(s_logits : alloc_if(0) free_if(0)) \
                in(matmul_selecting_mic)
            {
                void *matmul_mic;
                if (matmul_selecting_mic == 0) {
                    matmul_mic = matmul_naive_mic;
                } else if (matmul_selecting_mic == 1) {
                    matmul_mic = matmul_mkl_mic;
                } else if (matmul_selecting_mic == 2) {
                    matmul_mic = matmul_imci; 
                }

                forward_mic(
                    x, 
                    dim,
                    kv_dim,
                    kv_mul,
                    hidden_dim,
                    head_size,
                    p_n_layers, 
                    p_seq_len,
                    p_n_kv_heads,
                    p_n_heads,
                    p_dim, 
                    p_vocab_size,
                    w_rms_att_weight,
                    w_wq,
                    w_wk,
                    w_wv,
                    w_wo, 
                    w_rms_ffn_weight,
                    w_w1,
                    w_w2,
                    w_w3,
                    w_rms_final_weight,
                    w_wcls, 
                    s_q,
                    s_key_cache,
                    s_value_cache,
                    s_xb,
                    s_xb2,
                    s_hb,
                    s_hb2,
                    s_att, 
                    s_logits,
                    token, 
                    pos, 
                    0,
                    offloaded_layers,
                    matmul_mic
                );
            }
            #pragma offload_wait target(mic : 0) wait(sign)

            if (offloaded_layers < p_n_layers) {
                int *sign;
                // only transmit new KV$. 
				for (int l = 0; l < offloaded_layers; l++) {
                    // key and value point to the kv cache
                    int loff = l * p_seq_len * kv_dim; // kv cache layer offset for convenience
                    float *s_k = s_key_cache + loff + pos * kv_dim;
                    float *s_v = s_value_cache + loff + pos * kv_dim;

                    #pragma offload target(mic : 0) signal(sign) \
                        out(s_k : length(kv_dim) alloc_if(0) free_if(0)) \
                        out(s_v : length(kv_dim) alloc_if(0) free_if(0)) 
                    {}
                    //#pragma offload_wait target(mic : 0) wait(sign)
				}
                
                // only transmit new attentions
                for (int h = 0; h < p_n_heads; h++) {
                    float *s_att_h = s_att + h * p_seq_len; 
                    #pragma offload target(mic : 0) signal(sign) \
                        out(s_att_h : length(pos) alloc_if(0) free_if(0))
                    {}
                    #pragma offload_wait target(mic : 0) wait(sign)
                }
            }
        }
        // forward the transformer to get logits for the next token
        if (offloaded_layers < p_n_layers) {
            forward_cpu(transformer, token, pos, offloaded_layers, p->n_layers, matmul_cpu);
        } else {
            int *sign;
            #pragma offload target(mic : 0) signal(sign) \
                out(s_logits : length(p_vocab_size) alloc_if(0) free_if(0))
            {}
            #pragma offload_wait target(mic : 0) wait(sign)
        }
        float* logits = transformer->state.logits;
        next = sample(sampler, logits);
        pos++;



        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");

    // Free allocated memory on MIC
    if (offloaded_layers > 0) {
        char *sign;

        printf("Freeing on MIC\n");
        #pragma offload target(mic : 0) signal(sign) \
            nocopy(x : alloc_if(0) free_if(1)) \
            nocopy(w_rms_att_weight :alloc_if(0) free_if(1)) \
            nocopy(w_wq : alloc_if(0) free_if(1)) \
            nocopy(w_wk : alloc_if(0) free_if(1)) \
            nocopy(w_wv : alloc_if(0) free_if(1)) \
            nocopy(w_wo : alloc_if(0) free_if(1)) \
            nocopy(w_rms_ffn_weight : alloc_if(0) free_if(1)) \
            nocopy(w_w1 : alloc_if(0) free_if(1)) \
            nocopy(w_w2 : alloc_if(0) free_if(1)) \
            nocopy(w_w3 : alloc_if(0) free_if(1)) \
            nocopy(w_rms_final_weight : alloc_if(0) free_if(1)) \
            nocopy(w_wcls : alloc_if(0) free_if(1)) \
            nocopy(s_q : alloc_if(0) free_if(1)) \
            nocopy(s_key_cache : alloc_if(0) free_if(1)) \
            nocopy(s_value_cache : alloc_if(0) free_if(1)) \
            nocopy(s_xb : alloc_if(0) free_if(1)) \
            nocopy(s_xb2 : alloc_if(0) free_if(1)) \
            nocopy(s_hb : alloc_if(0) free_if(1)) \
            nocopy(s_hb2 : alloc_if(0) free_if(1)) \
            nocopy(s_att : alloc_if(0) free_if(1)) \
            nocopy(s_logits : alloc_if(0) free_if(1))
        {}
        #pragma offload_wait target(mic : 0) wait(sign)
        printf("Freeing on MIC done\n");
    }

    _mm_free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -o <int>    number of offloaded layers, -1 = ALL\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    fprintf(stderr, "  -M <int> choose Matmul function on CPU: 0:naive|1:mkl|2:avx2, default: 0\n");
    fprintf(stderr, "  -N <int> choose Matmul function on MIC: 0:naive|1:mkl|2:imci, default: 1\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    void *matmul_cpu = matmul_avx2;
    //void *matmul_mic = matmul_mkl_mic;
    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
    int offloaded_layers = -1;
    int matmul_selecting_cpu = 0;
	int matmul_selecting_mic = 2; 

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 'o') { offloaded_layers = atoi(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
		else if (argv[i][1] == 'M') { matmul_selecting_cpu = atoi(argv[i + 1]); }
		else if (argv[i][1] == 'N') { matmul_selecting_mic = atoi(argv[i + 1]); }
        else { error_usage(); }
    }

    if (matmul_selecting_cpu == 0) {
        matmul_cpu = matmul_naive;
        printf("Choosed naive matmul on CPU...\n");
    } else if (matmul_selecting_cpu == 1) {
        matmul_cpu = matmul_mkl;
        printf("Choosed MKL matmul on CPU...\n");
    } else {
        matmul_cpu = matmul_avx2;
        printf("Choosed AVX2 matmul on CPU...\n");
    }

    if (matmul_selecting_mic == 0) {
        printf("Choosed naive matmul on MIC...\n");
    } else if (matmul_selecting_mic == 1) {
        printf("Choosed MKL matmul on MIC...\n");
	} else if (matmul_selecting_mic == 2) { 
        printf("Choosed IMCI matmul on MIC...\n"); 
	}


    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

	if (offloaded_layers == -1 || offloaded_layers > transformer.config.n_layers) {
		offloaded_layers = transformer.config.n_layers;
	}

	printf("Tokenizer size: %d\n", transformer.config.vocab_size);
    printf("Transformer config:\n");
    printf("\tmodel name: %s\n", checkpoint_path);
    printf("\tn_layers: %d\n", transformer.config.n_layers);
    printf("\tn_heads: %d\n", transformer.config.n_heads);
    printf("\tn_kv_heads: %d\n", transformer.config.n_kv_heads);
    printf("\tdim: %d\n", transformer.config.dim);
    printf("\thidden_dim: %d\n", transformer.config.hidden_dim);
    printf("\tseq_len: %d\n", transformer.config.seq_len);
    printf("Offloading %d of %d layers to MIC\n", offloaded_layers, transformer.config.n_layers);
    printf("Steps limit: %d\n", steps); 

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps, offloaded_layers, matmul_cpu, matmul_selecting_mic);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps, offloaded_layers, matmul_cpu, matmul_selecting_mic);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif