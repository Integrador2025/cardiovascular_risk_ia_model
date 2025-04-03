from data_processing import preprocess_data
from training import train_with_stratified_kfold
import load

def main():
    df_pacientes, _, _ = load.load_dataset()
    X, Y, scaler, feature_names = preprocess_data(df_pacientes)
    
    if input("¿Entrenar modelo? (si/no): ").lower() in ('si', 'sí', 'yes', 'y'):
        train_with_stratified_kfold(X, Y, feature_names)
        print("Modelo entrenado!")
    else:
        print("Operación cancelada.")

if __name__ == '__main__':
    main()