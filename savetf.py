saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')




#else everything else remains same
