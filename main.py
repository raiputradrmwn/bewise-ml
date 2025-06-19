from flask import Flask, request, jsonify
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import joblib
app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"message": "NutriScore Fuzzy Logic API - Ready to calculate nutrition scores"})

# --------------------------
# Fuzzy Logic Setup
# --------------------------

# Beverage Antecedents
energy_bev = ctrl.Antecedent(np.arange(0, 401, 1), 'energy_bev')
saturated_fats_bev = ctrl.Antecedent(np.arange(0, 11, 0.1), 'saturated_fats_bev')
sugars_bev = ctrl.Antecedent(np.arange(0, 51, 0.1), 'sugars_bev')
sodium_bev = ctrl.Antecedent(np.arange(0, 5, 0.01), 'sodium_bev')
proteins_bev = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'proteins_bev')
fibers_bev = ctrl.Antecedent(np.arange(0, 10, 0.1), 'fibers_bev')
fruits_vegetables_bev = ctrl.Antecedent(np.arange(0, 81, 1), 'fruits_vegetables_bev')

# Food Antecedents
energy_food = ctrl.Antecedent(np.arange(0, 3501, 1), 'energy_food')
saturated_fats_food = ctrl.Antecedent(np.arange(0, 11, 0.1), 'saturated_fats_food')
sugars_food = ctrl.Antecedent(np.arange(0, 52, 0.1), 'sugars_food')
sodium_food = ctrl.Antecedent(np.arange(0, 5, 0.01), 'sodium_food')
proteins_food = ctrl.Antecedent(np.arange(0, 20, 0.1), 'proteins_food')
fibers_food = ctrl.Antecedent(np.arange(0, 10, 0.1), 'fibers_food')
fruits_vegetables_food = ctrl.Antecedent(np.arange(0, 101, 1), 'fruits_vegetables_food')

# Consequents
n_points_bev = ctrl.Consequent(np.arange(0, 41, 1), 'n_points_bev')
p_points_bev = ctrl.Consequent(np.arange(0, 16, 1), 'p_points_bev')
n_points_food = ctrl.Consequent(np.arange(0, 41, 1), 'n_points_food')
p_points_food = ctrl.Consequent(np.arange(0, 16, 1), 'p_points_food')

# --------------------------
# Beverage Membership Functions
# --------------------------

# Energy (kJ/100mL)
energy_bev['0'] = fuzz.trapmf(energy_bev.universe, [0, 0, 25, 30])
energy_bev['1'] = fuzz.trapmf(energy_bev.universe, [25, 30, 85, 90])
energy_bev['2'] = fuzz.trapmf(energy_bev.universe, [85, 90, 145, 150])
energy_bev['3'] = fuzz.trapmf(energy_bev.universe, [145, 150, 205, 210])
energy_bev['4'] = fuzz.trapmf(energy_bev.universe, [205, 210, 235, 240])
energy_bev['5'] = fuzz.trapmf(energy_bev.universe, [235, 240, 265, 270])
energy_bev['6'] = fuzz.trapmf(energy_bev.universe, [265, 270, 295, 300])
energy_bev['7'] = fuzz.trapmf(energy_bev.universe, [295, 300, 325, 330])
energy_bev['8'] = fuzz.trapmf(energy_bev.universe, [325, 330, 355, 360])
energy_bev['9'] = fuzz.trapmf(energy_bev.universe, [355, 360, 385, 390])
energy_bev['10'] = fuzz.trapmf(energy_bev.universe, [385, 390, 400, 400])

# Saturated Fats (g/100mL)
saturated_fats_bev['0'] = fuzz.trapmf(saturated_fats_bev.universe, [0, 0, 0.9, 1.0])
saturated_fats_bev['1'] = fuzz.trapmf(saturated_fats_bev.universe, [0.9, 1.0, 1.9, 2.0])
saturated_fats_bev['2'] = fuzz.trapmf(saturated_fats_bev.universe, [1.9, 2.0, 2.9, 3.0])
saturated_fats_bev['3'] = fuzz.trapmf(saturated_fats_bev.universe, [2.9, 3.0, 3.9, 4.0])
saturated_fats_bev['4'] = fuzz.trapmf(saturated_fats_bev.universe, [3.9, 4.0, 4.9, 5.0])
saturated_fats_bev['5'] = fuzz.trapmf(saturated_fats_bev.universe, [4.9, 5.0, 5.9, 6.0])
saturated_fats_bev['6'] = fuzz.trapmf(saturated_fats_bev.universe, [5.9, 6.0, 6.9, 7.0])
saturated_fats_bev['7'] = fuzz.trapmf(saturated_fats_bev.universe, [6.9, 7.0, 7.9, 8.0])
saturated_fats_bev['8'] = fuzz.trapmf(saturated_fats_bev.universe, [7.9, 8.0, 8.9, 9.0])
saturated_fats_bev['9'] = fuzz.trapmf(saturated_fats_bev.universe, [8.9, 9.0, 9.9, 10.0])
saturated_fats_bev['10'] = fuzz.trapmf(saturated_fats_bev.universe, [9.9, 10.0, 11.0, 11.0])

# Sugars (g/100mL)
sugars_bev['0'] = fuzz.trapmf(sugars_bev.universe, [0, 0, 0.4, 0.5])
sugars_bev['1'] = fuzz.trapmf(sugars_bev.universe, [0.4, 0.5, 1.9, 2])
sugars_bev['2'] = fuzz.trapmf(sugars_bev.universe, [1.9, 2, 3.4, 3.5])
sugars_bev['3'] = fuzz.trapmf(sugars_bev.universe, [3.4, 3.5, 4.9, 5])
sugars_bev['4'] = fuzz.trapmf(sugars_bev.universe, [4.9, 5, 5.9, 6])
sugars_bev['5'] = fuzz.trapmf(sugars_bev.universe, [5.9, 6, 6.9, 7])
sugars_bev['6'] = fuzz.trapmf(sugars_bev.universe, [6.9, 7, 7.9, 8])
sugars_bev['7'] = fuzz.trapmf(sugars_bev.universe, [7.9, 8, 8.9, 9])
sugars_bev['8'] = fuzz.trapmf(sugars_bev.universe, [8.9, 9, 9.9, 10])
sugars_bev['9'] = fuzz.trapmf(sugars_bev.universe, [9.9, 10, 10.9, 11])
sugars_bev['10'] = fuzz.trapmf(sugars_bev.universe, [10.9, 11, 49, 50])

# Sodium (g/100mL)
sodium_bev['0'] = fuzz.trapmf(sodium_bev.universe, [0, 0, 0.19, 0.2])
sodium_bev['1'] = fuzz.trapmf(sodium_bev.universe, [0.19, 0.2, 0.39, 0.4])
sodium_bev['2'] = fuzz.trapmf(sodium_bev.universe, [0.39, 0.4, 0.59, 0.6])
sodium_bev['3'] = fuzz.trapmf(sodium_bev.universe, [0.59, 0.6, 0.79, 0.8])
sodium_bev['4'] = fuzz.trapmf(sodium_bev.universe, [0.79, 0.8, 0.99, 1.0])
sodium_bev['5'] = fuzz.trapmf(sodium_bev.universe, [0.99, 1.0, 1.19, 1.2])
sodium_bev['6'] = fuzz.trapmf(sodium_bev.universe, [1.19, 1.2, 1.39, 1.4])
sodium_bev['7'] = fuzz.trapmf(sodium_bev.universe, [1.39, 1.4, 1.59, 1.6])
sodium_bev['8'] = fuzz.trapmf(sodium_bev.universe, [1.59, 1.6, 1.79, 1.8])
sodium_bev['9'] = fuzz.trapmf(sodium_bev.universe, [1.79, 1.8, 1.99, 2.0])
sodium_bev['10'] = fuzz.trapmf(sodium_bev.universe, [1.99, 2.0, 2.19, 2.2])
sodium_bev['11'] = fuzz.trapmf(sodium_bev.universe, [2.19, 2.2, 2.39, 2.4])
sodium_bev['12'] = fuzz.trapmf(sodium_bev.universe, [2.39, 2.4, 2.59, 2.6])
sodium_bev['13'] = fuzz.trapmf(sodium_bev.universe, [2.59, 2.6, 2.79, 2.8])
sodium_bev['14'] = fuzz.trapmf(sodium_bev.universe, [2.79, 2.8, 2.99, 3.0])
sodium_bev['15'] = fuzz.trapmf(sodium_bev.universe, [2.99, 3.0, 3.19, 3.2])
sodium_bev['16'] = fuzz.trapmf(sodium_bev.universe, [3.19, 3.2, 3.39, 3.4])
sodium_bev['17'] = fuzz.trapmf(sodium_bev.universe, [3.39, 3.4, 3.59, 3.6])
sodium_bev['18'] = fuzz.trapmf(sodium_bev.universe, [3.59, 3.6, 3.79, 3.8])
sodium_bev['19'] = fuzz.trapmf(sodium_bev.universe, [3.79, 3.8, 4.0, 4.0])
sodium_bev['20'] = fuzz.trapmf(sodium_bev.universe, [3.9, 4.0, 5.0, 5.0])

# Proteins (g/100mL)
proteins_bev['0'] = fuzz.trapmf(proteins_bev.universe, [0, 0, 1.1, 1.2])
proteins_bev['1'] = fuzz.trapmf(proteins_bev.universe, [1.1, 1.2, 1.4, 1.5])
proteins_bev['2'] = fuzz.trapmf(proteins_bev.universe, [1.4, 1.5, 1.7, 1.8])
proteins_bev['3'] = fuzz.trapmf(proteins_bev.universe, [1.7, 1.8, 2.0, 2.1])
proteins_bev['4'] = fuzz.trapmf(proteins_bev.universe, [2.0, 2.1, 2.3, 2.4])
proteins_bev['5'] = fuzz.trapmf(proteins_bev.universe, [2.3, 2.4, 2.6, 2.7])
proteins_bev['6'] = fuzz.trapmf(proteins_bev.universe, [2.6, 2.7, 2.9, 3.0])
proteins_bev['7'] = fuzz.trapmf(proteins_bev.universe, [2.9, 3.0, 9.9, 10.0])

# Fibers (g/100mL)
fibers_bev['0'] = fuzz.trapmf(fibers_bev.universe, [0, 0, 2.9, 3.0])
fibers_bev['1'] = fuzz.trapmf(fibers_bev.universe, [2.9, 3.0, 4.0, 4.1])
fibers_bev['2'] = fuzz.trapmf(fibers_bev.universe, [4.0, 4.1, 5.1, 5.2])
fibers_bev['3'] = fuzz.trapmf(fibers_bev.universe, [5.1, 5.2, 6.2, 6.3])
fibers_bev['4'] = fuzz.trapmf(fibers_bev.universe, [6.2, 6.3, 7.3, 7.4])
fibers_bev['5'] = fuzz.trapmf(fibers_bev.universe, [7.3, 7.4, 10, 10])

# Fruits/Vegetables (%)
fruits_vegetables_bev['0'] = fuzz.trapmf(fruits_vegetables_bev.universe, [0, 0, 39, 40])
fruits_vegetables_bev['2'] = fuzz.trapmf(fruits_vegetables_bev.universe, [39, 40, 59, 60])
fruits_vegetables_bev['4'] = fuzz.trapmf(fruits_vegetables_bev.universe, [59, 60, 79, 80])
fruits_vegetables_bev['5'] = fuzz.trapmf(fruits_vegetables_bev.universe, [79, 80, 100, 100])

# --------------------------
# Food Membership Functions
# --------------------------

# Energy (kJ/100g)
energy_food['0'] = fuzz.trapmf(energy_food.universe, [0, 0, 330, 335])
energy_food['1'] = fuzz.trapmf(energy_food.universe, [330, 335, 665, 670])
energy_food['2'] = fuzz.trapmf(energy_food.universe, [665, 670, 1000, 1005])
energy_food['3'] = fuzz.trapmf(energy_food.universe, [1000, 1005, 1335, 1340])
energy_food['4'] = fuzz.trapmf(energy_food.universe, [1335, 1340, 1670, 1675])
energy_food['5'] = fuzz.trapmf(energy_food.universe, [1670, 1675, 2005, 2010])
energy_food['6'] = fuzz.trapmf(energy_food.universe, [2005, 2010, 2340, 2345])
energy_food['7'] = fuzz.trapmf(energy_food.universe, [2340, 2345, 2675, 2680])
energy_food['8'] = fuzz.trapmf(energy_food.universe, [2675, 2680, 3010, 3015])
energy_food['9'] = fuzz.trapmf(energy_food.universe, [3010, 3015, 3345, 3350])
energy_food['10'] = fuzz.trapmf(energy_food.universe, [3345, 3350, 3500, 3500])

# Saturated Fats (g/100g)
saturated_fats_food['0'] = fuzz.trapmf(saturated_fats_food.universe, [0, 0, 0.9, 1.0])
saturated_fats_food['1'] = fuzz.trapmf(saturated_fats_food.universe, [0.9, 1.0, 1.9, 2.0])
saturated_fats_food['2'] = fuzz.trapmf(saturated_fats_food.universe, [1.9, 2.0, 2.9, 3.0])
saturated_fats_food['3'] = fuzz.trapmf(saturated_fats_food.universe, [2.9, 3.0, 3.9, 4.0])
saturated_fats_food['4'] = fuzz.trapmf(saturated_fats_food.universe, [3.9, 4.0, 4.9, 5.0])
saturated_fats_food['5'] = fuzz.trapmf(saturated_fats_food.universe, [4.9, 5.0, 5.9, 6.0])
saturated_fats_food['6'] = fuzz.trapmf(saturated_fats_food.universe, [5.9, 6.0, 6.9, 7.0])
saturated_fats_food['7'] = fuzz.trapmf(saturated_fats_food.universe, [6.9, 7.0, 7.9, 8.0])
saturated_fats_food['8'] = fuzz.trapmf(saturated_fats_food.universe, [7.9, 8.0, 8.9, 9.0])
saturated_fats_food['9'] = fuzz.trapmf(saturated_fats_food.universe, [8.9, 9.0, 9.9, 10.0])
saturated_fats_food['10'] = fuzz.trapmf(saturated_fats_food.universe, [9.9, 10.0, 20.9, 21.0])

# Sugars (g/100g)
sugars_food['0'] = fuzz.trapmf(sugars_food.universe, [0, 0, 3.3, 3.4])
sugars_food['1'] = fuzz.trapmf(sugars_food.universe, [3.3, 3.4, 6.7, 6.8])
sugars_food['2'] = fuzz.trapmf(sugars_food.universe, [6.7, 6.8, 9.9, 10.0])
sugars_food['3'] = fuzz.trapmf(sugars_food.universe, [9.9, 10.0, 13.9, 14.0])
sugars_food['4'] = fuzz.trapmf(sugars_food.universe, [13.9, 14.0, 16.9, 17.0])
sugars_food['5'] = fuzz.trapmf(sugars_food.universe, [16.9, 17.0, 19.9, 20.0])
sugars_food['6'] = fuzz.trapmf(sugars_food.universe, [19.9, 20.0, 23.9, 24.0])
sugars_food['7'] = fuzz.trapmf(sugars_food.universe, [23.9, 24.0, 26.9, 27.0])
sugars_food['8'] = fuzz.trapmf(sugars_food.universe, [26.9, 27.0, 30.9, 31.0])
sugars_food['9'] = fuzz.trapmf(sugars_food.universe, [30.9, 31.0, 36.9, 37.0])
sugars_food['10'] = fuzz.trapmf(sugars_food.universe, [36.9, 37.0, 40.9, 41.0])
sugars_food['11'] = fuzz.trapmf(sugars_food.universe, [40.9, 41.0, 43.9, 44.0])
sugars_food['12'] = fuzz.trapmf(sugars_food.universe, [43.9, 44.0, 47.9, 48.0])
sugars_food['13'] = fuzz.trapmf(sugars_food.universe, [47.9, 48.0, 50.9, 51.0])
sugars_food['14'] = fuzz.trapmf(sugars_food.universe, [50.9, 51.0, 52.0, 52.0])
sugars_food['15'] = fuzz.trapmf(sugars_food.universe, [51.0, 51.0, 52.0, 52.0])

# Sodium (g/100g)
sodium_food['0'] = fuzz.trapmf(sodium_food.universe, [0, 0, 0.19, 0.2])
sodium_food['1'] = fuzz.trapmf(sodium_food.universe, [0.19, 0.2, 0.39, 0.4])
sodium_food['2'] = fuzz.trapmf(sodium_food.universe, [0.39, 0.4, 0.59, 0.6])
sodium_food['3'] = fuzz.trapmf(sodium_food.universe, [0.59, 0.6, 0.79, 0.8])
sodium_food['4'] = fuzz.trapmf(sodium_food.universe, [0.79, 0.8, 0.99, 1.0])
sodium_food['5'] = fuzz.trapmf(sodium_food.universe, [0.99, 1.0, 1.19, 1.2])
sodium_food['6'] = fuzz.trapmf(sodium_food.universe, [1.19, 1.2, 1.39, 1.4])
sodium_food['7'] = fuzz.trapmf(sodium_food.universe, [1.39, 1.4, 1.59, 1.6])
sodium_food['8'] = fuzz.trapmf(sodium_food.universe, [1.59, 1.6, 1.79, 1.8])
sodium_food['9'] = fuzz.trapmf(sodium_food.universe, [1.79, 1.8, 1.99, 2.0])
sodium_food['10'] = fuzz.trapmf(sodium_food.universe, [1.99, 2.0, 2.19, 2.2])
sodium_food['11'] = fuzz.trapmf(sodium_food.universe, [2.19, 2.2, 2.39, 2.4])
sodium_food['12'] = fuzz.trapmf(sodium_food.universe, [2.39, 2.4, 2.59, 2.6])
sodium_food['13'] = fuzz.trapmf(sodium_food.universe, [2.59, 2.6, 2.79, 2.8])
sodium_food['14'] = fuzz.trapmf(sodium_food.universe, [2.79, 2.8, 2.99, 3.0])
sodium_food['15'] = fuzz.trapmf(sodium_food.universe, [2.99, 3.0, 3.19, 3.2])
sodium_food['16'] = fuzz.trapmf(sodium_food.universe, [3.19, 3.2, 3.39, 3.4])
sodium_food['17'] = fuzz.trapmf(sodium_food.universe, [3.39, 3.4, 3.59, 3.6])
sodium_food['18'] = fuzz.trapmf(sodium_food.universe, [3.59, 3.6, 3.79, 3.8])
sodium_food['19'] = fuzz.trapmf(sodium_food.universe, [3.79, 3.8, 4.0, 4.0])
sodium_food['20'] = fuzz.trapmf(sodium_food.universe, [3.9, 4.0, 5.0, 5.0])

# Proteins (g/100g)
proteins_food['0'] = fuzz.trapmf(proteins_food.universe, [0, 0, 1.9, 2.4])
proteins_food['1'] = fuzz.trapmf(proteins_food.universe, [1.9, 2.4, 4.8, 5.3])
proteins_food['2'] = fuzz.trapmf(proteins_food.universe, [4.8, 5.3, 7.2, 7.7])
proteins_food['3'] = fuzz.trapmf(proteins_food.universe, [7.2, 7.7, 9.6, 10.1])
proteins_food['4'] = fuzz.trapmf(proteins_food.universe, [9.6, 10.1, 12.0, 12.5])
proteins_food['5'] = fuzz.trapmf(proteins_food.universe, [12.0, 12.5, 14.0, 14.5])
proteins_food['6'] = fuzz.trapmf(proteins_food.universe, [14.0, 14.5, 17.0, 17.5])
proteins_food['7'] = fuzz.trapmf(proteins_food.universe, [17.0, 17.5, 20, 20])

# Fibers (g/100g)
fibers_food['0'] = fuzz.trapmf(fibers_food.universe, [0, 0, 2.9, 3.0])
fibers_food['1'] = fuzz.trapmf(fibers_food.universe, [2.9, 3.0, 4.0, 4.1])
fibers_food['2'] = fuzz.trapmf(fibers_food.universe, [4.0, 4.1, 5.1, 5.2])
fibers_food['3'] = fuzz.trapmf(fibers_food.universe, [5.1, 5.2, 6.2, 6.3])
fibers_food['4'] = fuzz.trapmf(fibers_food.universe, [6.2, 6.3, 7.3, 7.4])
fibers_food['5'] = fuzz.trapmf(fibers_food.universe, [7.3, 7.4, 10, 10])

# Fruits/Vegetables (%)
fruits_vegetables_food['0'] = fuzz.trapmf(fruits_vegetables_food.universe, [0, 0, 39, 40])
fruits_vegetables_food['1'] = fuzz.trapmf(fruits_vegetables_food.universe, [39, 40, 59, 60])
fruits_vegetables_food['2'] = fuzz.trapmf(fruits_vegetables_food.universe, [59, 60, 79, 80])
fruits_vegetables_food['5'] = fuzz.trapmf(fruits_vegetables_food.universe, [79, 80, 100, 100])

# --------------------------
# Output Membership Functions
# --------------------------

# Beverage Negative Points (0-40)
# Beverage Negative Points (0-40)
for i in range(21):
    a = max(0, i*2 - 1)
    b = i*2
    c = min(40, (i+1)*2)
    d = min(40, c + 1)
    if d < c:
        d = c
    n_points_bev[str(i)] = fuzz.trapmf(n_points_bev.universe, [a, b, c, d])

# Beverage Positive Points (0-15)
for i in range(8):
    a = max(0, i*2 - 1)
    b = i*2
    c = min(15, (i+1)*2)
    d = min(15, c + 1)
    if d < c:
        d = c
    p_points_bev[str(i)] = fuzz.trapmf(p_points_bev.universe, [a, b, c, d])

# Food Negative Points (0-40)
for i in range(21):
    a = max(0, i*2 - 1)
    b = i*2
    c = min(40, (i+1)*2)
    d = min(40, c + 1)
    if d < c:
        d = c
    n_points_food[str(i)] = fuzz.trapmf(n_points_food.universe, [a, b, c, d])

# Food Positive Points (0-15)
for i in range(8):
    a = max(0, i*2 - 1)
    b = i*2
    c = min(15, (i+1)*2)
    d = min(15, c + 1)
    if d < c:
        d = c
    p_points_food[str(i)] = fuzz.trapmf(p_points_food.universe, [a, b, c, d])


# --------------------------
# Rule Creation
# --------------------------

# Beverage Rules
bev_energy_rules = [ctrl.Rule(energy_bev[str(i)], n_points_bev[str(i)]) for i in range(11)]
bev_saturated_fats_rules = [ctrl.Rule(saturated_fats_bev[str(i)], n_points_bev[str(i)]) for i in range(11)]
bev_sugars_rules = [ctrl.Rule(sugars_bev[str(i)], n_points_bev[str(i)]) for i in range(11)]
bev_sodium_rules = [ctrl.Rule(sodium_bev[str(i)], n_points_bev[str(i)]) for i in range(21)]

bev_protein_rules = [ctrl.Rule(proteins_bev[str(i)], p_points_bev[str(i)]) for i in range(8)]
bev_fiber_rules = [ctrl.Rule(fibers_bev[str(i)], p_points_bev[str(i)]) for i in range(6)]
bev_fv_rules = [
    ctrl.Rule(fruits_vegetables_bev['0'], p_points_bev['0']),
    ctrl.Rule(fruits_vegetables_bev['2'], p_points_bev['2']),
    ctrl.Rule(fruits_vegetables_bev['4'], p_points_bev['4']),
    ctrl.Rule(fruits_vegetables_bev['5'], p_points_bev['5'])
]

# Food Rules
food_energy_rules = [ctrl.Rule(energy_food[str(i)], n_points_food[str(i)]) for i in range(11)]
food_saturated_fats_rules = [ctrl.Rule(saturated_fats_food[str(i)], n_points_food[str(i)]) for i in range(11)]
food_sugars_rules = [ctrl.Rule(sugars_food[str(i)], n_points_food[str(i)]) for i in range(16)]
food_sodium_rules = [ctrl.Rule(sodium_food[str(i)], n_points_food[str(i)]) for i in range(21)]

food_protein_rules = [ctrl.Rule(proteins_food[str(i)], p_points_food[str(i)]) for i in range(8)]
food_fiber_rules = [ctrl.Rule(fibers_food[str(i)], p_points_food[str(i)]) for i in range(6)]
food_fv_rules = [
    ctrl.Rule(fruits_vegetables_food['0'], p_points_food['0']),
    ctrl.Rule(fruits_vegetables_food['1'], p_points_food['1']),
    ctrl.Rule(fruits_vegetables_food['2'], p_points_food['2']),
    ctrl.Rule(fruits_vegetables_food['5'], p_points_food['5'])
]

# --------------------------
# Control Systems
# --------------------------

# Beverage Control Systems
bev_n_ctrl = ctrl.ControlSystem(
    bev_energy_rules + 
    bev_saturated_fats_rules + 
    bev_sugars_rules + 
    bev_sodium_rules
)

bev_p_ctrl = ctrl.ControlSystem(
    bev_protein_rules + 
    bev_fiber_rules + 
    bev_fv_rules
)

# Food Control Systems
food_n_ctrl = ctrl.ControlSystem(
    food_energy_rules + 
    food_saturated_fats_rules + 
    food_sugars_rules + 
    food_sodium_rules
)

food_p_ctrl = ctrl.ControlSystem(
    food_protein_rules + 
    food_fiber_rules + 
    food_fv_rules
)

# --------------------------
# Control System Simulations
# --------------------------

bev_n_scoring = ctrl.ControlSystemSimulation(bev_n_ctrl)
bev_p_scoring = ctrl.ControlSystemSimulation(bev_p_ctrl)
food_n_scoring = ctrl.ControlSystemSimulation(food_n_ctrl)
food_p_scoring = ctrl.ControlSystemSimulation(food_p_ctrl)

# --------------------------
# Calculation Functions
# --------------------------

def calculate_beverage_score(nutrition_data):
    try:
        # Convert energy to kJ if needed (assuming input is in kcal)
        energy_kj = nutrition_data['energy'] * 4.184
        
        # Set inputs for negative points
        bev_n_scoring.input['energy_bev'] = energy_kj
        bev_n_scoring.input['saturated_fats_bev'] = nutrition_data['saturated_fat']
        bev_n_scoring.input['sugars_bev'] = nutrition_data['sugar']
        bev_n_scoring.input['sodium_bev'] = nutrition_data['sodium']
        bev_n_scoring.compute()
        n_points = bev_n_scoring.output['n_points_bev']
        
        # Set inputs for positive points
        bev_p_scoring.input['proteins_bev'] = nutrition_data['protein']
        bev_p_scoring.input['fibers_bev'] = nutrition_data['fiber']
        bev_p_scoring.input['fruits_vegetables_bev'] = nutrition_data['fruit_vegetable']
        bev_p_scoring.compute()
        p_points = bev_p_scoring.output['p_points_bev']

        # Calculate final score
        nutri_score = n_points - p_points

        # SPECIAL CASE for water / zero nutrition (air mineral)
        input_is_water = (
            nutrition_data['energy'] <= 1e-4 and
            nutrition_data['saturated_fat'] <= 1e-4 and
            nutrition_data['sugar'] <= 1e-4 and
            nutrition_data['sodium'] <= 0.02 and # sodium still in 0.015 typical for air
            nutrition_data['protein'] <= 1e-4 and
            nutrition_data['fiber'] <= 1e-4 and
            nutrition_data['fruit_vegetable'] <= 1e-4
        )
        if input_is_water:
            return {
                "nutri_score": 0,
                "category": "1",
                "n_points": 0,
                "p_points": 0
            }
        if (
            nutrition_data['protein'] == 0 and
            nutrition_data['fiber'] == 0 and
            nutrition_data['fruit_vegetable'] == 0
        ):
            p_points = 0

        if nutri_score < 1.5:
            nutri_score = 0
            category = "1"
        elif nutri_score <= 2.5:
            category = "2"
        elif nutri_score <= 6.5:
            category = "3"
        elif nutri_score <= 9.5:
            category = "4"
        else:   
            category = "5"
            
        return {
            "nutri_score": round(nutri_score, 2),
            "category": category,
            "n_points": n_points,
            "p_points": p_points
        }
        
    except Exception as e:
        return {"error": str(e)}

    
def calculate_food_score(nutrition_data):
    """Calculate NutriScore for food using fuzzy logic"""
    try:
        # Convert energy to kJ if needed (assuming input is in kcal)
        energy_kj = nutrition_data['energy'] * 4.184
        
        # Set inputs for negative points
        food_n_scoring.input['energy_food'] = energy_kj
        food_n_scoring.input['saturated_fats_food'] = nutrition_data['saturated_fat']
        food_n_scoring.input['sugars_food'] = nutrition_data['sugar']
        food_n_scoring.input['sodium_food'] = nutrition_data['sodium']
        food_n_scoring.compute()
        n_points = food_n_scoring.output['n_points_food']
        
        # Set inputs for positive points
        food_p_scoring.input['proteins_food'] = nutrition_data['protein']
        food_p_scoring.input['fibers_food'] = nutrition_data['fiber']
        food_p_scoring.input['fruits_vegetables_food'] = nutrition_data['fruit_vegetable']
        food_p_scoring.compute()
        p_points = food_p_scoring.output['p_points_food']
        
        # Calculate final score
        nutri_score = n_points - p_points
        
        # Determine category
        if nutri_score <= 0.5:
            category = "1"
        elif nutri_score <= 2.5:
            category = "2"
        elif nutri_score <= 10.5:
            category = "3"
        elif nutri_score <= 18.5:
            category = "4"
        else:
            category = "5"
            
        return {
            "p_points" : p_points,
            "n_points": n_points,
            "nutri_score": round(nutri_score, 2),
            "category": category
        }
        
    except Exception as e:
        return {"error": str(e)}
    
# --------------------------
# API Endpoints
# --------------------------

@app.route('/calculate-nutri-score/food', methods=['POST'])
def calculate_food():
    try:
        data_list = request.json
        
        if not isinstance(data_list, list):
            return jsonify({"error": "Input must be a list of food products"}), 400
            
        results = []
        for data in data_list:
            if 'nutritionFact' in data:
                nutrition_data = data['nutritionFact']
            else:
                return jsonify({"error": "Each food product must contain 'nutritionFact'"}), 400
            result = calculate_food_score(nutrition_data)
            results.append(result)
            
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/calculate-nutri-score/beverages', methods=['POST'])
def calculate_beverages():
    try:
        data_list = request.json
        
        if not isinstance(data_list, list):
            return jsonify({"error": "Input must be a list of beverages"}), 400
            
        results = []
        for data in data_list:
            if 'nutritionFact' in data:
                nutrition_data = data['nutritionFact']
            else:
                return jsonify({"error": "Each beverage must contain 'nutritionFact'"}), 400
            result = calculate_beverage_score(nutrition_data)
            results.append(result)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400
model = joblib.load("bewise_nutriscore_modelfix.pkl")
FEATURES = ['energy', 'saturated_fat', 'sugar', 'sodium', 'protein', 'fiber', 'fruit_vegetable']
@app.route('/predict-nutriscore', methods=['POST'])
def predict_nutriscore():
    data = request.json

    try:
        feats = data.get("features", data)
        X = np.array([[feats[feat] for feat in FEATURES]])
    except Exception as e:
        return jsonify({"error": f"Invalid input, required features: {FEATURES}. Error: {e}"}), 400
    try:
        y_pred = model.predict(X)
        nutriscore_label = y_pred[0] 
        return jsonify({"nutriscore_label": str(nutriscore_label)})
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

if __name__ == "__main__":
    app.run(port=8080)