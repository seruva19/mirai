/* Stable C boundary for liburing's opaque userspace structures. */
#include <errno.h>
#include <liburing.h>
#include <stdint.h>
#include <stdlib.h>

struct mirai_ring {
    struct io_uring ring;
    int ready;
};

int mirai_uring_open(void **result)
{
    struct mirai_ring *state = calloc(1, sizeof(*state));
    if (state == NULL)
        return -ENOMEM;
    int rc = io_uring_queue_init(2, &state->ring, 0);
    if (rc < 0) {
        free(state);
        return rc;
    }
    state->ready = 1;
    *result = state;
    return 0;
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

void mirai_uring_close(void *handle)
{
    struct mirai_ring *state = handle;
    if (state->ready)
        io_uring_queue_exit(&state->ring);
    free(state);
}
