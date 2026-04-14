FROM alpine:latest
RUN apk add --no-cache gcc musl-dev python3 strace
COPY syscall_exec.c /syscall_exec.c
RUN gcc -O2 -static -o /syscall_exec /syscall_exec.c
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["/syscall_exec"]
