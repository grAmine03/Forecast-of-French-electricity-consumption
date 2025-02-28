from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
import pandas as pd
import joblib
import os
from Models import *  # Assuming Models.py contains your MLP_Model definition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_test = pd.read_csv("final_test.csv")

# Load scalers
scaler_X = joblib.load("scaler_X.joblib")
scaler_y = joblib.load("scaler_y.joblib")


# Load test data

categorical_features = ['month', 'day', 'dayofweek', 'saison']

# Encode categorical features
enc = OneHotEncoder(handle_unknown='ignore')
encoded_features = enc.fit_transform(final_test[categorical_features]).toarray()
encoded_df = pd.DataFrame(encoded_features, columns=enc.get_feature_names_out(categorical_features))

# Merge encoded data
X_test_new = pd.concat([final_test, encoded_df], axis=1).drop(columns=categorical_features)

model = MLP_Batch_Model(input_dim=X_test_new.shape[1], output_dim=25)
#model = CNNModel(output_dim=25)  
model.load_state_dict(torch.load("MLP_batch_Model.pth", map_location=device))
model.to(device)
model.eval()

# Normalize test data
X_test_new = scaler_X.fit_transform(X_test_new)
X_test_new = torch.tensor(X_test_new, dtype=torch.float32).to(device) 
# Ensure the input_dim is correct
input_dim = X_test_new.shape[1]
print(f"Input dimension: {input_dim}")
model = MLP_Model(input_dim=input_dim, output_dim=25)

model.load_state_dict(torch.load("MLP_Model.pth", map_location=device))
model.to(device)
model.eval()

# Make predictions
with torch.no_grad():
    y_pred_new = model(X_test_new).cpu().numpy()

# Inverse transform predictions
y_pred_new = scaler_y.inverse_transform(y_pred_new)

# Save predictions
pred = pd.read_csv("pred.csv")  # Replace with actual file
target_columns = ['France', 'Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne',
                  'Centre-Val de Loire', 'Grand Est', 'Hauts-de-France', 'Normandie',
                  'Nouvelle-Aquitaine', 'Occitanie', 'Pays de la Loire',
                  "Provence-Alpes-Côte d'Azur", 'Île-de-France',
                  'Montpellier Méditerranée Métropole', 'Métropole Européenne de Lille',
                  'Métropole Grenoble-Alpes-Métropole', "Métropole Nice Côte d'Azur",
                  'Métropole Rennes Métropole', 'Métropole Rouen Normandie',
                  "Métropole d'Aix-Marseille-Provence", 'Métropole de Lyon',
                  'Métropole du Grand Nancy', 'Métropole du Grand Paris',
                  'Nantes Métropole', 'Toulouse Métropole']

for i, col_name in enumerate(target_columns):
    pred[f'pred_{col_name}'] = y_pred_new[:, i]

# Save final predictions
pred.to_csv('pred_avec_predictions.csv', index=False)
print("Predictions saved to 'pred_avec_predictions.csv'.")
# Charger le fichier original
pred = pd.read_csv("pred.csv")  # Remplace par le bon chemin

# Assurez-vous que y_pred_new a la bonne forme (même nombre de colonnes que target_columns)
y_pred_df = pd.DataFrame(y_pred_new, columns=target_columns)

# Ajouter toutes les prédictions au DataFrame `pred`
for col in target_columns:
    pred[f'pred_{col}'] = y_pred_df[col]

# Sauvegarde en CSV
pred.to_csv("pred2.csv", index=False)

# Vérification
if os.path.exists("pred2.csv"):
    print("✅ Le fichier 'pred2.csv' a été créé avec succès !")
else:
    print("❌ Erreur : le fichier 'pred2.csv' n'a pas été créé.")
