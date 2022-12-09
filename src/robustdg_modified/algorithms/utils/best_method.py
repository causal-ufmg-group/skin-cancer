import numpy as np


def get_best_method(validation_acc, final_acc):

    best_method = np.argmax(validation_acc)

    return {
        "epoch": best_method,
        "Validation Acc": validation_acc[best_method],
        "Test Acc": final_acc[best_method],
    }
