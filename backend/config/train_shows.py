out_dir = 'out-shows'
eval_interval = 250 # keep frequent because of overfit
eval_iters = 200
log_interval = 10 # don't print too too often

always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shows'
wandb_run_name = 'mini-gpt'

dataset = 'shows'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99

warmup_iters = 100 