class Config:
    DATA_PATH = "./data/"
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    HIDDEN_DIM = 64
    NUM_ATTENTION_HEADS = 8
    TRANSFORMER_LAYERS = 3
    GRU_LAYERS = 1
    TEMPERATURE_C = 0.07  
    TEMPERATURE_V = 0.1   
    TEMPERATURE_S = 0.1   
    DWA_TEMPERATURE = 4   
    
    # training
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 20
    WEIGHT_DECAY = 1e-5
    
    # YOLO
    YOLO_MODEL = "yolov8"
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    
    DEVICE = "cuda"
    SEED = 42
