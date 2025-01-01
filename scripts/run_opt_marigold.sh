# Set CUDA_VISIBLE_DEVICES to 0
export CUDA_VISIBLE_DEVICES=0

# Get marigold initial affine-invariant sharp structure depth guidance
python run_marigold.py --input_root_dir ./examples/3

echo "Structure guidance is generated."

# Run the test-time alignment with sparse depth
python run_opt_marigold.py --input_root_dir ./examples/3 --r_ssim_depth