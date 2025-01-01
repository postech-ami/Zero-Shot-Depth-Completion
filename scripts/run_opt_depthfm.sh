# Set CUDA_VISIBLE_DEVICES to 0
export CUDA_VISIBLE_DEVICES=0

# DEPTHFM_CKPT_PATH="./pretrained_models/depthfm.pth"
DEPTHFM_CKPT_PATH="/node_data/hyoseok/checkpoints/depthfm-v1.ckpt"
INPUT_DIR="./examples/3"

# Get depthfm initial affine-invariant sharp structure depth guidance
python run_depthfm.py --checkpoint "$DEPTHFM_CKPT_PATH" --input_root_dir "$INPUT_DIR"

echo "Structure guidance is generated."

# Run the test-time alignment with sparse depth guidance
python run_opt_depthfm.py --checkpoint "$DEPTHFM_CKPT_PATH" --input_root_dir "$INPUT_DIR" --r_ssim_depth # --n_inter 1
