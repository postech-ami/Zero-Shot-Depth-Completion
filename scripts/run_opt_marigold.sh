# Set CUDA_VISIBLE_DEVICES to 0
export CUDA_VISIBLE_DEVICES=0

INPUT_DIR="./examples/3"

# Get marigold initial affine-invariant sharp structure depth guidance
python run_marigold.py --input_root_dir "$INPUT_DIR"

echo "Structure guidance is generated."

# Run the test-time alignment with sparse depth
python run_opt_marigold.py --input_root_dir "$INPUT_DIR" --r_ssim_depth