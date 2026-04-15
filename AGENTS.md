# Running scripts in bash

When running scripts in the terminal, use `tmux` to allow the user to monitor the progress of the script. Attach to the tmux session after launching it to monitor the progress yourself:

```bash
tmux new-session -d -s <session-name> <script>
tmux capture-pane -t <session-name> -p
```

> Note: if `tmux` is not installed, fall back to normal bash and suggest the user to install it
