# Revert CUDA paths to that as before activation
if [ -n "$OLD_LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
fi

