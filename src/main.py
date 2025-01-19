from train_model import train_model
from evaluate_model import evaluate_model

def main():
    # Train the model
    model, history = train_model()

    #Evaluate model
    evaluate_model()

if __name__=='__main__':
    main()
