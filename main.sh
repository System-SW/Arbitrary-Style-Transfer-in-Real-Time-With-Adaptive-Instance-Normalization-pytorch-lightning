python main.py \
--device="cuda:0" \
--content_root_dir="{YOUR CONTENT DATA PATH}" \
--style_root_dir="{YOUR STYLE DATA PATH}" \
--num_workers=8 \
--batch_size=8 \
--log_every_n_steps=5 \
--gpus=1 \
--max_steps=160000