class Language:
    FRENCH = "FR"

class DefaultConfigurations:
    FILTER_SIZES = "3,4,5"
    NUM_FILTERS = 128
    DROPOUT_KEEP_PROB = 0.85
    L2_REG_LAMBDA = 1e-5
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    DISPLAY_EVERY = 20
    EVALUATE_EVERY = 100
    NUM_CHECKPOINTS = 5
    LEARNING_RATE = 1e-3
    ALLOW_SOFT_PLACEMENT = True
    LOG_DEVICE_PLACEMENT = False
    GPU_ALLOW_GROWTH = True

    SAVE_SUMMARIES = False # For Tensorboard

if __name__ == '__main__':
    configs = DefaultConfigurations("FR")




