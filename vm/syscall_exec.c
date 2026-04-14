/*
 * syscall_exec: executes raw syscalls received over TCP.
 *
 * Listens on port 7777. For each connection:
 *   - Reads syscall requests (binary protocol)
 *   - Executes via syscall()
 *   - Writes back the result
 *
 * Protocol (binary, little-endian):
 *   Request:
 *     uint16_t  syscall_nr
 *     uint16_t  buf_len
 *     int64_t   arg1..arg6  (48 bytes)
 *     uint8_t   buffer[buf_len]
 *   Response:
 *     int64_t   retval
 *     int32_t   errno_val
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <netinet/in.h>
#include <fcntl.h>

#define PORT 7777
#define BUF_OFFSET_BASE 0x10000
#define MAX_BUF 4096
#define WORK_DIR "/workspace"

static uint8_t buf[MAX_BUF];

/* Read exactly n bytes from fd */
static int readn(int fd, void *dst, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, (char*)dst + got, n - got);
        if (r <= 0) return -1;
        got += r;
    }
    return 0;
}

/* Write exactly n bytes to fd */
static int writen(int fd, const void *src, size_t n) {
    size_t sent = 0;
    while (sent < n) {
        ssize_t w = write(fd, (const char*)src + sent, n - sent);
        if (w <= 0) return -1;
        sent += w;
    }
    return 0;
}

/* Resolve syscall arg: if it looks like a buffer reference, point into buf */
static long resolve_arg(int64_t arg, uint8_t *buffer, uint16_t buf_len) {
    if (arg >= BUF_OFFSET_BASE && arg < BUF_OFFSET_BASE + buf_len) {
        return (long)(buffer + (arg - BUF_OFFSET_BASE));
    }
    return (long)arg;
}

static void handle_client(int client_fd) {
    while (1) {
        uint16_t syscall_nr, buf_len;
        int64_t args[6];

        if (readn(client_fd, &syscall_nr, 2) < 0) break;
        if (readn(client_fd, &buf_len, 2) < 0) break;
        if (readn(client_fd, args, 48) < 0) break;

        if (buf_len > MAX_BUF) buf_len = MAX_BUF;
        if (buf_len > 0) {
            if (readn(client_fd, buf, buf_len) < 0) break;
        }

        /* Resolve pointer arguments */
        long a[6];
        for (int i = 0; i < 6; i++) {
            a[i] = resolve_arg(args[i], buf, buf_len);
        }

        fprintf(stderr, "syscall nr=%d buf_len=%d args=[%ld,%ld,%ld,%ld,%ld,%ld]\n",
                syscall_nr, buf_len, a[0], a[1], a[2], a[3], a[4], a[5]);

        /* Execute */
        errno = 0;
        long ret = syscall(syscall_nr, a[0], a[1], a[2], a[3], a[4], a[5]);
        int err = errno;

        fprintf(stderr, "  -> ret=%ld errno=%d\n", ret, err);

        /* Send response */
        int64_t retval = ret;
        int32_t errno_val = err;
        if (writen(client_fd, &retval, 8) < 0) break;
        if (writen(client_fd, &errno_val, 4) < 0) break;
    }
}

int main(void) {
    mkdir(WORK_DIR, 0755);
    chdir(WORK_DIR);

    setvbuf(stderr, NULL, _IONBF, 0);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(PORT),
        .sin_addr.s_addr = INADDR_ANY,
    };

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); return 1;
    }
    if (listen(server_fd, 1) < 0) {
        perror("listen"); return 1;
    }

    fprintf(stderr, "syscall_exec: listening on port %d\n", PORT);

    while (1) {
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) { perror("accept"); continue; }

        /* Move server and client fds out of the low range (3-9)
           so that openat/dup/etc return low fds as expected */
        int high_client = dup2(client_fd, 100);
        close(client_fd);
        close(server_fd);

        fprintf(stderr, "syscall_exec: client connected (fd=%d)\n", high_client);
        handle_client(high_client);
        close(high_client);
        fprintf(stderr, "syscall_exec: client disconnected\n");
    }

    return 0;
}
