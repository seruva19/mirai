/* Stable C boundary for liburing's opaque userspace structures. */
#define _GNU_SOURCE
#include <errno.h>
#include <liburing.h>
#include <linux/mman.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>

#define MIRAI_FIXED_BYTES (2U * 1024U * 1024U)

struct mirai_ring {
    struct io_uring ring;
    int ready;
    void *fixed_buffer;
};

static int open_ring(void **result, int fixed, int sqpoll, int hugepages)
{
    struct mirai_ring *state = calloc(1, sizeof(*state));
    if (state == NULL)
        return -ENOMEM;
    struct io_uring_params params = {0};
    if (sqpoll) {
        params.flags = IORING_SETUP_SQPOLL;
        params.sq_thread_idle = 1000;
    }
    int rc = io_uring_queue_init_params(2, &state->ring, &params);
    if (rc < 0) {
        free(state);
        return rc;
    }
    state->ready = 1;
    if (sqpoll && !(params.features & IORING_FEAT_SQPOLL_NONFIXED)) {
        io_uring_queue_exit(&state->ring);
        free(state);
        return -EOPNOTSUPP;
    }
    if (fixed) {
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
        if (hugepages)
            flags |= MAP_HUGETLB | MAP_HUGE_2MB;
        void *buffer = mmap(NULL, MIRAI_FIXED_BYTES, PROT_READ | PROT_WRITE, flags, -1, 0);
        if (buffer == MAP_FAILED) {
            rc = -errno;
        } else {
            struct iovec iov = {.iov_base = buffer, .iov_len = MIRAI_FIXED_BYTES};
            rc = io_uring_register_buffers(&state->ring, &iov, 1);
            if (rc < 0)
                munmap(buffer, MIRAI_FIXED_BYTES);
            else
                state->fixed_buffer = buffer;
        }
        if (rc < 0) {
            io_uring_queue_exit(&state->ring);
            free(state);
            return rc;
        }
    }
    *result = state;
    return 0;
}

int mirai_uring_open(void **result)
{
    return open_ring(result, 0, 0, 0);
}

int mirai_uring_open_fixed(void **result, int sqpoll, int hugepages)
{
    return open_ring(result, 1, sqpoll, hugepages);
}

void *mirai_uring_fixed_buffer(void *handle)
{
    return ((struct mirai_ring *)handle)->fixed_buffer;
}

static int submit_and_wait(struct mirai_ring *state)
{
    int rc = io_uring_submit(&state->ring);
    if (rc == 1) {
        struct io_uring_cqe *completion;
        do {
            rc = io_uring_wait_cqe(&state->ring, &completion);
        } while (rc == -EINTR);
        if (rc == 0) {
            rc = completion->res;
            io_uring_cqe_seen(&state->ring, completion);
            return rc;
        }
    }
    io_uring_queue_exit(&state->ring);
    state->ready = 0;
    return rc < 0 ? rc : -EIO;
}

int mirai_uring_read(void *handle, int fd, void *buffer, unsigned length,
                    uint64_t offset)
{
    struct mirai_ring *state = handle;
    if (!state->ready)
        return -EIO;
    struct io_uring_sqe *sqe = io_uring_get_sqe(&state->ring);
    if (sqe == NULL)
        return -EBUSY;
    io_uring_prep_read(sqe, fd, buffer, length, offset);
    return submit_and_wait(state);
}

int mirai_uring_read_fixed(void *handle, int fd, unsigned length, uint64_t offset)
{
    struct mirai_ring *state = handle;
    if (!state->ready)
        return -EIO;
    if (state->fixed_buffer == NULL || length > MIRAI_FIXED_BYTES)
        return -EINVAL;
    struct io_uring_sqe *sqe = io_uring_get_sqe(&state->ring);
    if (sqe == NULL)
        return -EBUSY;
    io_uring_prep_read_fixed(sqe, fd, state->fixed_buffer, length, offset, 0);
    return submit_and_wait(state);
}

void mirai_uring_close(void *handle)
{
    struct mirai_ring *state = handle;
    if (state->ready)
        io_uring_queue_exit(&state->ring);
    if (state->fixed_buffer != NULL)
        munmap(state->fixed_buffer, MIRAI_FIXED_BYTES);
    free(state);
}
