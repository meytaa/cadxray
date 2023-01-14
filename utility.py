
def target_to_oh(target, NUM_CLASS):
    one_hot = [0] * NUM_CLASS
    one_hot[target] = 1
    return one_hot