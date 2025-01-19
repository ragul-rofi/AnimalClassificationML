from tensorflow.keras.models import load_model
from load_data import load_data

def evaluate_model():
    #Loading data

    _,_,test_data = load_data()

    #loading the best model(assuming 'best_model.h5' is saved after trainng)
    model = load_model('best_model.keras')

    #evaluate the model on hte test data

    test_loss, test_accuracy = model.evaluate(test_data)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100: .2f}%")