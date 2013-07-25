/*
 * A simple restricted transactional memory micro-benchmark.
 *
 * Copyright (c) 2013 Carl G. Ritson <cgr@kent.ac.uk>
 *
 * This file may be freely used, copied, or distributed without compensation 
 * or licensing restrictions, but is done so without any warranty or 
 * implication of merchantability or fitness for any particular purpose.
 *
 * gcc -Wall -O2 rtm-bench.c -o rtm-bench -lpthread -lrt
 */

#define _GNU_SOURCE

#ifdef __APPLE__
#define AFFINITY 0
#else
#define AFFINITY 1
#define HAS_AFFINITY 1
#define HAS_CLOCK 1
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

#include <sys/mman.h>
#include <sys/time.h>
#include <sched.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#include "rtm.h"

// memory constants
#define CACHELINE_BYTES         (64)
#if __LP64__
#define MEM_BASE                (0x600000000000ULL)
#else
#define MEM_BASE                (0x60000000UL)
#endif

// timing data
typedef struct _timing_t {
    uint64_t start_tsc, end_tsc;
    uint64_t start, end;
    uint64_t elapsed_ns;
    uint64_t elapsed_cycles;
} timing_t;

// log structure
#define LOG_BUFSIZE 1024
#define LOG_DATASIZE (128 * 1024)
typedef struct _thread_log_t {
    int enabled;
    unsigned int pos;
    unsigned int size;
    char data[LOG_DATASIZE];
} thread_log_t;

// default configuration
static unsigned int  config_max_threads         = 8;
static unsigned long config_thread_memory_size  = 512 * 1024 * 1024;
static unsigned long config_thread_gap_size     = 512 * 1024 * 1024;
static unsigned long config_op_max_size         = 32 * 1024;
static unsigned long config_op_max_cycles       = 128 * 1024;
static unsigned long config_test_loops          = 1;
static unsigned int  config_test                = 0;
static unsigned int  config_thread_shifting     = 1;
static unsigned int  config_isolated_tests      = 1;
static unsigned int  config_shared_tests        = 1;
static unsigned int  config_limited_tests       = 1;

// thread data
typedef struct _thread_param_t {
    int type;
} thread_param_t;

static pthread_t *threads = NULL;
static int n_threads = 0;
static thread_param_t *thread_params = NULL;
static thread_log_t **thread_logs = NULL;
static pthread_mutex_t thread_log_lock;

static pthread_mutex_t barrier_mutex;
static pthread_cond_t barrier_condition;
static volatile int barrier_count;

// shared memory
static void *memory_ptr = NULL;
static unsigned long memory_size = 0;
static int use_shared_memory = 0;
static void **thread_memory = NULL;
static unsigned long thread_memory_size = 0;

// test constants
typedef void *(*test_thread_t)(void *); 
#define SUCCESS     (42)
#define FAILURE     (41)
#define MAGIC       (42)
#define CAS_P       (MAGIC + 1)
#define CAS_Q       (MAGIC * 2)
#define N_COUNTERS  (SUCCESS + 1)
enum thread_type_t {
    U_READ      = 1,
    U_WRITE     = 2,
    U_CAS       = 3,
    X_READ      = 4,
    X_WRITE     = 5,
    X_CAS       = 6,
    X_ABORTN    = 7,
    X_ABORTM    = 8,
    N_TESTS     = 8
};

/*
 * logging functions
 */
static void flush_thread_log(const int id)
{
    thread_log_t *log = thread_logs[id];
    if (log == NULL)
        return;
    if (log->pos > 0) {
        pthread_mutex_lock(&thread_log_lock);
        fwrite(log->data, log->pos, 1, stdout);
        log->pos = 0;
        pthread_mutex_unlock(&thread_log_lock);
    }
}
static void flush_thread_logs(void)
{
    int i;

    if (thread_logs == NULL)
        return;

    for (i = 0; i < n_threads; ++i) {
        flush_thread_log(i);
    }
}

static void log_buffer_on(const int id)
{
    thread_logs[id]->enabled = 1;
}
static void log_buffer_off(const int id)
{
    thread_logs[id]->enabled = 0;
    flush_thread_log(id);
}

static void _thread_log(const int id, const char *msg, va_list ap)
{
    char buffer[LOG_BUFSIZE];
    int pos = 0, r;
    if (id >= 0) {
        r = snprintf(buffer + pos, LOG_BUFSIZE - pos, "%02d: ", id);
        pos = (pos + r >= LOG_BUFSIZE ? LOG_BUFSIZE - 1 : pos + r);
    }
    r = vsnprintf(buffer + pos, LOG_BUFSIZE - pos, msg, ap);
    pos = (pos + r >= LOG_BUFSIZE ? LOG_BUFSIZE - 1 : pos + r);
    r = snprintf(buffer + pos, LOG_BUFSIZE - pos, "\n");
    pos = (pos + r >= LOG_BUFSIZE ? LOG_BUFSIZE - 1 : pos + r);

    // see if we should store this log entry
    if (id >= 0 && thread_logs != NULL) {
        if (thread_logs[id] != NULL) {
            thread_log_t *log = thread_logs[id];
            if (log->enabled) {
                if (pos < (log->size - log->pos)) {
                    memcpy(((char *)log->data) + log->pos, buffer, pos);
                    log->pos += pos;
                } else {
                    // lossing log entry
                }
                return;
            }
        }
    }

    // not setup to store; output to console
    fwrite(buffer, pos, 1, stdout);
}

static void thread_log(const int id, const char *msg, ...)
{
    va_list ap;
    va_start(ap, msg);
    _thread_log(id, msg, ap);
    va_end(ap);
}

static void main_log(const char *msg, ...)
{
    va_list ap;
    va_start(ap, msg);
    _thread_log(-1, msg, ap);
    va_end(ap);
}

static void error_out(const char *msg, ...)
{
    va_list ap;
    va_start(ap, msg);

    flush_thread_logs();

    fprintf(stderr, "error: ");
    vfprintf(stderr, msg, ap);
    fprintf(stderr, "\n");
    
    va_end(ap);
    exit(1);
}

static void setup_log(const int id)
{
    thread_log_t *log = (thread_log_t *) malloc(sizeof(thread_log_t));
    log->enabled = 0;
    log->pos = 0;
    log->size = LOG_DATASIZE; 
    memset(log->data, 0, log->size);
    thread_logs[id] = log;
}

/* 
 * timing functions
 */
static inline uint64_t rdtsc(void)
{
    uint32_t lo, hi;
    uint64_t v;
    asm volatile ("rdtsc" : "=a" (lo), "=d" (hi));
    v = hi;
    v <<= 32;
    v |= lo;
    return v;
}

static inline uint64_t get_time_ns(void)
{
    #ifdef HAS_CLOCK
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
    #else /* !HAS_CLOCK */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000000ULL) + (tv.tv_usec * 1000ULL);
    #endif /* !HAS_CLOCK */
}

static inline void timing_start(timing_t *t)
{
    t->start = get_time_ns();
    t->start_tsc = rdtsc();
}

static void timing_end(timing_t *t)
{
    t->end_tsc = rdtsc();
    t->end = get_time_ns();

    t->elapsed_ns = (t->end - t->start);
    t->elapsed_cycles = (t->end_tsc - t->start_tsc);
}

/*
 * thread functions
 */
static void init_threading(const int max_threads)
{
    int i;
    
    // thread state
    threads = (pthread_t *) malloc(sizeof(pthread_t) * max_threads);
    pthread_mutex_init(&thread_log_lock, NULL);
    thread_params = (thread_param_t *) malloc(sizeof(thread_param_t) * max_threads);
    thread_logs = (thread_log_t **) malloc(sizeof(thread_log_t *) * max_threads);
    thread_memory = (void **) malloc(sizeof(void *) * max_threads);
    for (i = 0; i < max_threads; ++i) {
        memset(&(thread_params[i]), 0, sizeof(thread_param_t));
        thread_logs[i] = NULL;
        thread_memory[i] = NULL;
    }
    
    // barrier
    pthread_mutex_init(&barrier_mutex, NULL);
    pthread_cond_init(&barrier_condition, NULL);
    barrier_count = 0;
}

static void wait_for_threads(void)
{
    int count = n_threads;
    int i;

    for (i = 0; i < count; ++i) {
        void *ret;
        pthread_join(threads[i], &ret);
        //main_log("joined thread %d, ret: %d", i, (long) ret);
    }
    n_threads = 0;

    for (i = 0; i < count; ++i) {
        free (thread_logs[i]);
        thread_logs[i] = NULL;
    }
}

static int start_thread(test_thread_t thread_main, unsigned int type)
{
    int id = n_threads;
    n_threads++;
    setup_log(id);
    thread_params[id].type = type;
    return pthread_create(&(threads[id]), NULL, thread_main, (void *) (long) id);
}

static void set_affinity(const int id)
{
    #ifdef HAS_AFFINITY
    cpu_set_t set;
    int ret;

    CPU_ZERO(&set);
    CPU_SET(id, &set);

    ret = sched_setaffinity(0, sizeof(set), &set);
    if (ret < 0) {
        thread_log(id, "error setting affinity, ret: %d, errno: %d\n", ret, errno);
    }
    
    ret = sched_getaffinity(0, sizeof(set), &set);
    if (ret < 0) {
        thread_log(id, "error getting affinity, ret: %d, errno: %d\n", ret, errno);
    }
    #else /* !HAS_AFFINITY */
    thread_log(id, "affinity not supported");
    #endif /* !HAS_AFFINITY */
}

static void set_barrier_count(const int n_threads)
{
    pthread_mutex_lock(&barrier_mutex);
    barrier_count = n_threads;
    pthread_mutex_unlock(&barrier_mutex);
}

static void barrier(void)
{
    pthread_mutex_lock(&barrier_mutex);
    barrier_count -= 1;
    if (barrier_count == 0) {
        barrier_count = n_threads;
        pthread_cond_broadcast(&barrier_condition);
    } else {
        pthread_cond_wait(&barrier_condition, &barrier_mutex);
    }
    pthread_mutex_unlock(&barrier_mutex);
}

static void boot_thread(const int id, int shared_mem, uint8_t **mem, unsigned long *mem_size)
{
    set_affinity(id);
    
    // pick memory
    if (shared_mem) {
        *mem = thread_memory[0];
    } else {
        *mem = thread_memory[id];
    }
    *mem_size = thread_memory_size;

    // warm up cache
    memset(*mem, 0, *mem_size);

    // initial barrier
    barrier();
}

static void shutdown_thread(const int id)
{
    n_threads -= 1;
    barrier();
}

/*
 * memory management
 */
static void setup_memory(const int max_threads,
        const unsigned long thread_length, 
        const unsigned long gap_length)
{
    unsigned int i;
    //int ret;
    
    memory_size = (max_threads * thread_length) 
                    + ((max_threads - 1) * gap_length); 
    main_log("allocating memory %llu bytes", memory_size); 
    memory_ptr = mmap((void *)MEM_BASE, memory_size, 
            PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, 0, 0);

    if (memory_ptr == ((char *)(-1))) {
        memory_ptr = NULL;
        memory_size = 0;
    }

    if (!memory_ptr) {
        error_out("failed to allocate memory, errno: %d", errno);
    }

    main_log("initialising memory: start"); 
    memset(memory_ptr, 0, memory_size);
    main_log("initialising memory: finish"); 
    /*
    ret = mlock(memory_ptr, memory_size);
    if (ret) {
        main_log("warning, failed to lock memory (need root?), errno: %d", errno);
    } 
    */

    // layout memory at fixed addresses with gaps
    for (i = 0; i < max_threads; ++i) {
        thread_memory[i] = (void *)((unsigned long)memory_ptr) + (i * (thread_length + gap_length));
    }
    thread_memory_size = thread_length;
}

static unsigned long compute_op_cycles(unsigned long op_size)
{
    unsigned long ideal_op_cycles = config_thread_memory_size / (op_size + CACHELINE_BYTES);
    if (ideal_op_cycles > config_op_max_cycles)
        return config_op_max_cycles;
    return ideal_op_cycles;
}

/* 
 * test operations
 */

static void memset32(uint8_t *_mem, uint32_t v, const unsigned long _size)
{
    volatile uint32_t *mem = (volatile uint32_t *)_mem;
    unsigned count = _size >> 2;
    while (count--)
        *(mem++) = v;
}

static void memset64(uint8_t *_mem, uint64_t v, const unsigned long _size)
{
    volatile uint64_t *mem = (volatile uint64_t *)_mem;
    unsigned count = _size >> 3;
    while (count--)
        *(mem++) = v;
}

// returns 1 on successful CAS
static inline uint32_t asm_lock_cas32(
    volatile uint32_t *ptr, 
    uint32_t ov, uint32_t nv)
{
    uint32_t result;

    asm volatile (
            "   lock; cmpxchg %3,(%1)\n"
            "   setz %%al           \n"
            "   and $1, %%eax      \n"
            : "=a" (result)
            : "r" (ptr), "0" (ov), "r" (nv)
            : "cc", "memory"
    );

    return result;
}

// returns 1 on successful CAS
static inline uint64_t asm_lock_cas64(
    volatile uint64_t *ptr, 
    uint64_t ov, uint64_t nv)
{
    uint64_t result;

    asm volatile (
            "   lock; cmpxchg %3,(%1)\n"
            "   setz %%al           \n"
            "   and $1, %%rax      \n"
            : "=A" (result)
            : "R" (ptr), "0" (ov), "R" (nv)
            : "cc", "memory"
    );

    return result;
}

static inline unsigned u_cas32(uint8_t *_mem, const unsigned long _size)
{
    volatile uint32_t *mem = (volatile uint32_t *)_mem;
    unsigned count = _size >> 2;
    unsigned ret = 0;
    
    while (count--) {
        ret |= (asm_lock_cas32(mem, CAS_P, CAS_Q) ^ 1);
        mem++;
    }
    ret = ret ? FAILURE : SUCCESS;
    
    return ret;
}

static inline unsigned u_cas64(uint8_t *_mem, const unsigned long _size)
{
    volatile uint64_t *mem = (volatile uint64_t *)_mem;
    unsigned count = _size >> 3;
    unsigned ret = 0;
    
    while (count--) {
        ret |= (asm_lock_cas64(mem, CAS_P, CAS_Q) ^ 1);
        mem++;
    }
    ret = ret ? FAILURE : SUCCESS;
    
    return ret;
}

static inline unsigned x_cas32(uint8_t *_mem, const unsigned long _size)
{
    volatile uint32_t *mem = (volatile uint32_t *)_mem;
    unsigned count = _size >> 2;
    unsigned ret = 0;
    
    ret = _xbegin();
    if (ret == _XBEGIN_STARTED) {
        ret = 0;
        while (count--) {
            if (*mem == CAS_P)
                *mem = CAS_Q;
            else
                ret = 1;
            mem++;
        }
        _xend();
        ret = ret ? FAILURE : SUCCESS;
    }
    
    return ret;
}

static inline unsigned x_cas64(uint8_t *_mem, const unsigned long _size)
{
    volatile uint64_t *mem = (volatile uint64_t *)_mem;
    unsigned count = _size >> 3;
    unsigned ret = 0;
    
    ret = _xbegin();
    if (ret == _XBEGIN_STARTED) {
        ret = 0;
        while (count--) {
            if (*mem == CAS_P)
                *mem = CAS_Q;
            else
                ret = 1;
            mem++;
        }
        _xend();
        ret = ret ? FAILURE : SUCCESS;
    }
    
    return ret;
}

static inline unsigned u_read32(uint8_t *_mem, const unsigned long _size)
{
    volatile uint32_t *mem = (volatile uint32_t *)_mem;
    unsigned count = _size >> 2;
    unsigned ret;
    
    while (count--) {
        ret += *mem;
        mem++;
    }
    ret = SUCCESS;
    
    return ret;   
}

static inline unsigned u_read64(uint8_t *_mem, const unsigned long _size)
{
    volatile uint64_t *mem = (volatile uint64_t *)_mem;
    unsigned count = _size >> 3;
    unsigned ret;
    
    while (count--) {
        ret += *mem;
        mem++;
    }
    ret = SUCCESS;
    
    return ret;   
}

static inline unsigned x_read32(uint8_t *_mem, const unsigned long _size)
{
    volatile uint32_t *mem = (volatile uint32_t *)_mem;
    unsigned count = _size >> 2;
    unsigned ret;
    
    ret = _xbegin();
    if (ret == _XBEGIN_STARTED) { 
        while (count--) {
            ret += *mem;
            mem++;
        }
        _xend();
        ret = SUCCESS;
    }
    
    return ret;   
}

static inline unsigned x_read64(uint8_t *_mem, const unsigned long _size)
{
    volatile uint64_t *mem = (volatile uint64_t *)_mem;
    unsigned count = _size >> 3;
    unsigned ret;
    
    ret = _xbegin();
    if (ret == _XBEGIN_STARTED) { 
        while (count--) {
            ret += *mem;
            mem++;
        }
        _xend();
        ret = SUCCESS;
    }
    
    return ret;   
}

static inline unsigned u_write32(uint8_t *_mem, const unsigned long _size)
{
    volatile uint32_t *mem = (volatile uint32_t *)_mem;
    unsigned count = _size >> 2;
    unsigned ret;
    
    while (count--) {
        *mem = MAGIC;
        mem++;
    }
    ret = SUCCESS;
    
    return ret;   
}

static inline unsigned u_write64(uint8_t *_mem, const unsigned long _size)
{
    volatile uint64_t *mem = (volatile uint64_t *)_mem;
    unsigned count = _size >> 3;
    unsigned ret;
    
    while (count--) {
        *mem = MAGIC;
        mem++;
    }
    ret = SUCCESS;
    
    return ret; 
}

static inline unsigned x_write32(uint8_t *_mem, const unsigned long _size)
{
    volatile uint32_t *mem = (volatile uint32_t *)_mem;
    unsigned count = _size >> 2;
    unsigned ret;
    
    ret = _xbegin();
    if (ret == _XBEGIN_STARTED) { 
        while (count--) {
            *mem = MAGIC;
            mem++;
        }
        _xend();
        ret = SUCCESS;
    }
    
    return ret;   
}

static inline unsigned x_write64(uint8_t *_mem, const unsigned long _size)
{
    volatile uint64_t *mem = (volatile uint64_t *)_mem;
    unsigned count = _size >> 3;
    unsigned ret;
    
    ret = _xbegin();
    if (ret == _XBEGIN_STARTED) { 
        while (count--) {
            *mem = MAGIC;
            mem++;
        }
        _xend();
        ret = SUCCESS;
    }
    
    return ret; 
}

static inline unsigned x_abort32(uint8_t *_mem, const unsigned long _size, unsigned long n)
{
    volatile uint32_t *mem = (volatile uint32_t *)_mem;
    unsigned count = _size >> 2;
    unsigned ret;
    
    ret = _xbegin();
    if (ret == _XBEGIN_STARTED) { 
        while (count--) {
            if (!(n--))
                _xabort(0);
            *mem = MAGIC;
            mem++;
        }
        if (!n)
            _xabort(0);
        _xend();
        ret = SUCCESS;
    }
    
    return ret;   
}
static inline unsigned x_abortn32(uint8_t *_mem, const unsigned long _size)
{
    return x_abort32(_mem, _size, 0);
}
static inline unsigned x_abortm32(uint8_t *_mem, const unsigned long _size)
{
    return x_abort32(_mem, _size, (_size >> 2));
}

static inline unsigned x_abort64(uint8_t *_mem, const unsigned long _size, unsigned long n)
{
    volatile uint64_t *mem = (volatile uint64_t *)_mem;
    unsigned count = _size >> 3;
    unsigned ret;
    
    ret = _xbegin();
    if (ret == _XBEGIN_STARTED) { 
        while (count--) {
            if (!(n--))
                _xabort(0);
            *mem = MAGIC;
            mem++;
        }
        if (!n)
            _xabort(0);
        _xend();
        ret = SUCCESS;
    }
    
    return ret; 
}
static inline unsigned x_abortn64(uint8_t *_mem, const unsigned long _size)
{
    return x_abort64(_mem, _size, 0);
}
static inline unsigned x_abortm64(uint8_t *_mem, const unsigned long _size)
{
    return x_abort64(_mem, _size, (_size >> 3));
}

/*
 * test harness
 */

static void *sleeper(void *param)
{
    const int id = (long) param;
    uint8_t *mem;
    unsigned long mem_size;
    boot_thread(id, use_shared_memory, &mem, &mem_size);
    thread_log(id, "sleeper");
    shutdown_thread(id);
    return NULL;
}

static inline void run_test(const int id, 
        const char *label,
        uint8_t *mem, const unsigned long mem_size,
        unsigned (*op)(uint8_t *, const unsigned long), 
        const unsigned long _count, const unsigned long op_size)
{
    //const unsigned long mask = (mem_size - 1);
    unsigned long stride;
    unsigned long counter[N_COUNTERS];
    unsigned long count = _count;
    timing_t t;
    uint8_t *ptr;
    int i;

    for (i = 0; i < N_COUNTERS; ++i)
        counter[i] = 0;
    
    stride = (op_size + (CACHELINE_BYTES - 1)) & (~(CACHELINE_BYTES - 1));
    ptr = mem;

    thread_log(id, "test = %s, count = %lu, op_size = %lu, stride = %lu",
        label, count, op_size, stride);

    barrier();

    timing_start(&t);
    while (count--) {
        unsigned ret = op(ptr, op_size);
        counter[ret]++;
        ptr += stride;
        /* could wrap, but count is limited to prevent overflow
        ptr = (uint8_t *)((((unsigned long)ptr) & (~mask)) | 
                    ((((unsigned long)ptr) + stride) & mask));
        */
    }
    timing_end(&t);

    barrier();

    thread_log(id, "ns = %llu, cycles = %llu", t.elapsed_ns, t.elapsed_cycles);
    for (i = 0; i < N_COUNTERS; ++i) {
        if (counter[i] > 0)
            thread_log(id, "counter %d = %lu", i, counter[i]);
    }

    barrier();
}

static void *test_thread(void *param)
{
    const int id = (long) param;
    uint8_t *mem;
    unsigned long mem_size;
    unsigned int i;
    
    boot_thread(id, use_shared_memory, &mem, &mem_size);
    
    for (i = 0; i < config_test_loops; ++i) {
        unsigned int n;

        switch (thread_params[id].type) {
            case U_READ:
                thread_log(id, "u_read");
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 4 : CACHELINE_BYTES)) 
                    run_test(id, "u_read32", mem, mem_size, u_read32, compute_op_cycles(n), n);
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 8 : CACHELINE_BYTES)) 
                    run_test(id, "u_read64", mem, mem_size, u_read64, compute_op_cycles(n), n);
                break;
    
            case U_WRITE:
                thread_log(id, "u_write");
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 4 : CACHELINE_BYTES)) 
                    run_test(id, "u_write32", mem, mem_size, u_write32, compute_op_cycles(n), n);
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 8 : CACHELINE_BYTES)) 
                    run_test(id, "u_write64", mem, mem_size, u_write64, compute_op_cycles(n), n);
                break;

            case U_CAS:
                thread_log(id, "u_cas");
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 4 : CACHELINE_BYTES)) { 
                    memset32(mem, CAS_P, mem_size);
                    run_test(id, "u_cas32", mem, mem_size, u_cas32, compute_op_cycles(n), n);
                }
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 8 : CACHELINE_BYTES)) {
                    memset64(mem, CAS_P, mem_size);
                    run_test(id, "u_cas64", mem, mem_size, u_cas64, compute_op_cycles(n), n);
                }
                break;
   
            case X_READ:
                thread_log(id, "x_read");
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 4 : CACHELINE_BYTES)) 
                    run_test(id, "x_read32", mem, mem_size, x_read32, compute_op_cycles(n), n);
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 8 : CACHELINE_BYTES)) 
                    run_test(id, "x_read64", mem, mem_size, x_read64, compute_op_cycles(n), n);
                break;

            case X_WRITE:
                thread_log(id, "x_write");
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 4 : CACHELINE_BYTES)) 
                    run_test(id, "x_write32", mem, mem_size, x_write32, compute_op_cycles(n), n);
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 8 : CACHELINE_BYTES)) 
                    run_test(id, "x_write64", mem, mem_size, x_write64, compute_op_cycles(n), n);
                break;
    
            case X_CAS:
                thread_log(id, "x_cas");
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 4 : CACHELINE_BYTES)) { 
                    memset32(mem, CAS_P, mem_size);
                    run_test(id, "x_cas32", mem, mem_size, x_cas32, compute_op_cycles(n), n);
                }
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 8 : CACHELINE_BYTES)) { 
                    memset64(mem, CAS_P, mem_size);
                    run_test(id, "x_cas64", mem, mem_size, x_cas64, compute_op_cycles(n), n);
                }
                break;
            
            case X_ABORTN:
                thread_log(id, "x_abortn");
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 4 : CACHELINE_BYTES)) { 
                    run_test(id, "x_abortn32", mem, mem_size, x_abortn32, compute_op_cycles(n), n);
                }
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 8 : CACHELINE_BYTES)) { 
                    run_test(id, "x_abortn64", mem, mem_size, x_abortn64, compute_op_cycles(n), n);
                }
                break;
            
            case X_ABORTM:
                thread_log(id, "x_abortm");
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 4 : CACHELINE_BYTES)) { 
                    run_test(id, "x_abortm32", mem, mem_size, x_abortm32, compute_op_cycles(n), n);
                }
                for (n = 0; n < config_op_max_size; n += (n < 1024 ? 8 : CACHELINE_BYTES)) { 
                    run_test(id, "x_abortm64", mem, mem_size, x_abortm64, compute_op_cycles(n), n);
                }
                break;
        }

        barrier();
    }

    shutdown_thread(id);
    return NULL;
}

/*
 * main
 */
static void usage(char *name)
{
    fprintf(stderr, 
        "Usage: %s [-m <bytes>] [-g <bytes>] [-c <cycles>] [-t <test-number>]\n"
        "\n"
        "  -m <bytes>   thread memory size           [default: %lu]\n"
        "  -g <bytes>   gap between threads          [default: %lu]\n"
        "  -c <cycles>  memory operation max. cycles [default: %lu]\n"
        "  -o <bytes>   memory operation max. size   [default: %lu]\n"
        "  -t <number>  run a specific test          [default is all]\n"
        "  -l <number>  number of test loops         [default: %lu]\n"
        "  -z <number>  override max threads         [default: %d]\n"
        "  -T           disable thread shifting\n"
        "  -I           disable isolated memory tests\n"
        "  -S           disable shared memory tests\n"
        "  -x           enable limited thread test program\n"
        "\n",
        name,
        config_thread_memory_size,
        config_thread_gap_size,
        config_op_max_cycles,
        config_op_max_size,
        config_test_loops,
        config_max_threads
    );
    exit(2);
}

static unsigned long ensure_pow2(unsigned long x)
{
    unsigned long y = x;
    int msb = 0;
    
    if (x == 0)
        return x;

    while (y) {
        y >>= 1;
        msb++;
    }

    y = 1 << (msb - 1);
    if (((y - 1) & x) == x)
        return x;
    else
        return y;
}

static void parse_args(int argc, char *argv[])
{
    char ch;
    while ((ch = getopt(argc, argv, "m:g:c:o:t:l:z:TISx")) != -1) {
        switch(ch) {
            case 'm':
                config_thread_memory_size = strtol(optarg, NULL, 10);
                config_thread_memory_size = ensure_pow2(config_thread_memory_size);
                break;
            case 'g':
                config_thread_gap_size = strtol(optarg, NULL, 10);
                config_thread_gap_size = ensure_pow2(config_thread_gap_size);
                break;
            case 'c':
                config_op_max_cycles = strtol(optarg, NULL, 10);
                break;
            case 'o':
                config_op_max_size = strtol(optarg, NULL, 10);
                break;
            case 't':
                config_test = strtol(optarg, NULL, 10);
                break;
            case 'l':
                config_test_loops = strtol(optarg, NULL, 10);
                break;
            case 'z':
                config_max_threads = strtol(optarg, NULL, 10);
                break;
            case 'T':
                config_thread_shifting = 0;
                break;
            case 'I':
                config_isolated_tests = 0;
                break;
            case 'S':
                config_shared_tests = 0;
                break;
            case 'x':
                config_limited_tests = 1;
                break;
            case '?':
            default:
                usage(argv[0]);
        }
    }
}

static int can_run_test(const int n)
{
    return (config_test == n || config_test <= 0);
}

static void run_single_thread_tests(void)
{
    int i, j, n;
    
    main_log("single thread tests");

    for (n = 0; n < (config_thread_shifting ? config_max_threads : 1); ++n) {
        for (i = 1; i <= N_TESTS; ++i) {
            if (!can_run_test(i))
                continue;

            set_barrier_count(n + 1);
            for (j = 0; j < n; ++j)
                start_thread(sleeper, 0);
            start_thread(test_thread, i);
            wait_for_threads();
        }
    }
}

static void run_homogenous_thread_tests(const int n_threads)
{
    int i, j, n;

    main_log("homogenous thread tests");
    
    for (n = 2; n <= n_threads; ++n) {
        for (i = 1; i <= N_TESTS; ++i) {
            if (!can_run_test(i))
                continue;
            
            set_barrier_count(n);
            for (j = 0; j < n; ++j)
                start_thread(test_thread, i);
            wait_for_threads();
        }
    }
}

static void run_heterogenous_thread_tests(void)
{
    int i, j;

    main_log("heterogenous thread tests");

    for (i = 1; i <= N_TESTS; ++i) {
        if (!can_run_test(i))
            continue;

        for (j = 1; j <= N_TESTS; ++j) {
            if (!can_run_test(j))
                continue;
            
            set_barrier_count(2);
            start_thread(test_thread, i);
            start_thread(test_thread, j);
            wait_for_threads();
        }
    }
}

static void isolated_memory(void)
{
    main_log("isolated memory tests");
    use_shared_memory = 0;
}

static void shared_memory(void)
{
    main_log("shared memory tests");
    use_shared_memory = 1;
}

static void run_limited_thread_tests(void)
{
    shared_memory();

    main_log("homogenous thread tests");

    set_barrier_count(2);
    start_thread(test_thread, X_READ);
    start_thread(test_thread, X_READ);
    wait_for_threads();
    
    set_barrier_count(2);
    start_thread(test_thread, X_WRITE);
    start_thread(test_thread, X_WRITE);
    wait_for_threads();
    
    set_barrier_count(2);
    start_thread(test_thread, U_CAS);
    start_thread(test_thread, U_CAS);
    wait_for_threads();
    
    set_barrier_count(2);
    start_thread(test_thread, X_CAS);
    start_thread(test_thread, X_CAS);
    wait_for_threads();


    main_log("heterogenous thread tests");

    set_barrier_count(2);
    start_thread(test_thread, X_CAS);
    start_thread(test_thread, U_READ);
    wait_for_threads();
    
    set_barrier_count(2);
    start_thread(test_thread, X_WRITE);
    start_thread(test_thread, U_WRITE);
    wait_for_threads();
}

int main(int argc, char *argv[])
{
    // setup
    config_max_threads = sysconf(_SC_NPROCESSORS_ONLN); 
    config_max_threads /= 2;
    
    parse_args(argc, argv);

    init_threading(config_max_threads);
    setup_memory(config_max_threads, config_thread_memory_size, config_thread_gap_size);
   
    // run tests
    if (config_limited_tests) {
        run_limited_thread_tests();
    } else {
        run_single_thread_tests();
        
        if (config_isolated_tests) {
            isolated_memory();
            run_homogenous_thread_tests(config_max_threads);
            run_heterogenous_thread_tests();
        }

        if (config_shared_tests) {
            shared_memory();
            run_homogenous_thread_tests(config_max_threads);
            run_heterogenous_thread_tests();
        }
    }
    
    // FIXME: tidy up

    return 0;
}
