from flask import Flask, request, jsonify
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Initialize Flask app
app = Flask(__name__)
@app.route('/')
def index():
    return jsonify({"Bewise": "Empower Smarter Nutrition Choices"})
# Define input variables food
energy = ctrl.Antecedent(np.arange(0, 3351, 1), 'energy')
saturated_fats = ctrl.Antecedent(np.arange(0, 11, 0.1), 'saturated_fats')
sugars = ctrl.Antecedent(np.arange(0, 52, 0.1), 'sugars')
sodium = ctrl.Antecedent(np.arange(0, 901, 1), 'sodium')
proteins = ctrl.Antecedent(np.arange(0, 18, 0.1), 'proteins')
fibers = ctrl.Antecedent(np.arange(0, 8, 0.1), 'fibers')
fruits_vegetables = ctrl.Antecedent(np.arange(0, 101, 1), 'fruits_vegetables')

# Define input variables for beverages
energy_beverages = ctrl.Antecedent(np.arange(0, 391, 1), 'energy_beverages')
saturated_fats_beverages = ctrl.Antecedent(np.arange(0, 11, 0.1), 'saturated_fats_beverages')
sugars_beverages = ctrl.Antecedent(np.arange(0, 20, 0.1), 'sugars_beverages')
sodium_beverages = ctrl.Antecedent(np.arange(0, 5, 0.1), 'sodium_beverages')
proteins_beverages = ctrl.Antecedent(np.arange(0, 4.5, 0.1), 'proteins_beverages')
fibers_beverages = ctrl.Antecedent(np.arange(0, 7.5, 0.1), 'fibers_beverages')
fruits_vegetables_beverages = ctrl.Antecedent(np.arange(0, 81, 1), 'fruits_vegetables_beverages')

# Define output variables
n_points_food = ctrl.Consequent(np.arange(0, 55, 1), 'n_points_food')
n_points_beverages = ctrl.Consequent(np.arange(0, 50, 1), 'n_points_beverages')
p_points_food = ctrl.Consequent(np.arange(0, 16, 1), 'p_points_food')
p_points_beverages = ctrl.Consequent(np.arange(0, 18, 1), 'p_points_beverages')

# Define membership functions for input variables food
energy['0'] = fuzz.trimf(energy.universe, [0, 0, 335])
energy['1'] = fuzz.trimf(energy.universe, [335, 335, 670])
energy['2'] = fuzz.trimf(energy.universe, [670, 670, 1005])
energy['3'] = fuzz.trimf(energy.universe, [1005, 1005, 1340])
energy['4'] = fuzz.trimf(energy.universe, [1340, 1340, 1675])
energy['5'] = fuzz.trimf(energy.universe, [1675, 1675, 2010])
energy['6'] = fuzz.trimf(energy.universe, [2010, 2010, 2345])
energy['7'] = fuzz.trimf(energy.universe, [2345, 2345, 2680])
energy['8'] = fuzz.trimf(energy.universe, [2680, 2680, 3015])
energy['9'] = fuzz.trimf(energy.universe, [3015, 3015, 3350])
energy['10'] = fuzz.trimf(energy.universe, [3350, 3350, 3351])

saturated_fats['0'] = fuzz.trimf(saturated_fats.universe, [0, 0, 1])
saturated_fats['1'] = fuzz.trimf(saturated_fats.universe, [1, 1, 2])
saturated_fats['2'] = fuzz.trimf(saturated_fats.universe, [2, 2, 3])
saturated_fats['3'] = fuzz.trimf(saturated_fats.universe, [3, 3, 4])
saturated_fats['4'] = fuzz.trimf(saturated_fats.universe, [4, 4, 5])
saturated_fats['5'] = fuzz.trimf(saturated_fats.universe, [5, 5, 6])
saturated_fats['6'] = fuzz.trimf(saturated_fats.universe, [6, 6, 7])
saturated_fats['7'] = fuzz.trimf(saturated_fats.universe, [7, 7, 8])
saturated_fats['8'] = fuzz.trimf(saturated_fats.universe, [8, 8, 9])
saturated_fats['9'] = fuzz.trimf(saturated_fats.universe, [9, 9, 10])
saturated_fats['10'] = fuzz.trimf(saturated_fats.universe, [10, 10, 11])

sugars['0'] = fuzz.trimf(sugars.universe, [0, 0, 3.4])
sugars['1'] = fuzz.trimf(sugars.universe, [3.4, 3.4, 6.8])
sugars['2'] = fuzz.trimf(sugars.universe, [6.8, 6.8, 10])
sugars['3'] = fuzz.trimf(sugars.universe, [10, 10, 14])
sugars['4'] = fuzz.trimf(sugars.universe, [14, 14, 17])
sugars['5'] = fuzz.trimf(sugars.universe, [17, 17, 20])
sugars['6'] = fuzz.trimf(sugars.universe, [20, 20, 24])
sugars['7'] = fuzz.trimf(sugars.universe, [24, 24, 27])
sugars['8'] = fuzz.trimf(sugars.universe, [27, 27, 31])
sugars['9'] = fuzz.trimf(sugars.universe, [31, 31, 37])
sugars['10'] = fuzz.trimf(sugars.universe, [37, 37, 52])

sodium['0'] = fuzz.trimf(sodium.universe, [0, 0, 90])
sodium['1'] = fuzz.trimf(sodium.universe, [90, 90, 180])
sodium['2'] = fuzz.trimf(sodium.universe, [180, 180, 270])
sodium['3'] = fuzz.trimf(sodium.universe, [270, 270, 360])
sodium['4'] = fuzz.trimf(sodium.universe, [360, 360, 450])
sodium['5'] = fuzz.trimf(sodium.universe, [450, 450, 540])
sodium['6'] = fuzz.trimf(sodium.universe, [540, 540, 630])
sodium['7'] = fuzz.trimf(sodium.universe, [630, 630, 720])
sodium['8'] = fuzz.trimf(sodium.universe, [720, 720, 810])
sodium['9'] = fuzz.trimf(sodium.universe, [810, 810, 900])
sodium['10'] = fuzz.trimf(sodium.universe, [900, 900, 901])

proteins['0'] = fuzz.trimf(proteins.universe, [0, 0, 2.4])
proteins['1'] = fuzz.trimf(proteins.universe, [2.4, 2.4, 4.8])
proteins['2'] = fuzz.trimf(proteins.universe, [4.8, 4.8, 7.2])
proteins['3'] = fuzz.trimf(proteins.universe, [7.2, 7.2, 9.6])
proteins['4'] = fuzz.trimf(proteins.universe, [9.6, 9.6, 12])
proteins['5'] = fuzz.trimf(proteins.universe, [12, 12, 14])
proteins['6'] = fuzz.trimf(proteins.universe, [14, 14, 17])
proteins['7'] = fuzz.trimf(proteins.universe, [17, 17, 18])

fibers['0'] = fuzz.trimf(fibers.universe, [0, 0, 3.0])
fibers['1'] = fuzz.trimf(fibers.universe, [3.0, 3.0, 4.1])
fibers['2'] = fuzz.trimf(fibers.universe, [4.1, 4.1, 5.2])
fibers['3'] = fuzz.trimf(fibers.universe, [5.2, 5.2, 6.3])
fibers['4'] = fuzz.trimf(fibers.universe, [6.3, 6.3, 7.4])
fibers['5'] = fuzz.trimf(fibers.universe, [7.4, 7.4, 8])

fruits_vegetables['0'] = fuzz.trimf(fruits_vegetables.universe, [0, 0, 40])
fruits_vegetables['1'] = fuzz.trimf(fruits_vegetables.universe, [40, 40, 60])
fruits_vegetables['2'] = fuzz.trimf(fruits_vegetables.universe, [60, 60, 80])
fruits_vegetables['5'] = fuzz.trimf(fruits_vegetables.universe, [80, 80, 101])

# Define membership functions for input variables Beverages
energy_beverages['0'] = fuzz.trimf(energy_beverages.universe, [0, 0, 30])
energy_beverages['1'] = fuzz.trimf(energy_beverages.universe, [30, 30, 90])
energy_beverages['2'] = fuzz.trimf(energy_beverages.universe, [90, 90, 150])
energy_beverages['3'] = fuzz.trimf(energy_beverages.universe, [150, 150, 210])
energy_beverages['4'] = fuzz.trimf(energy_beverages.universe, [210, 210, 240])
energy_beverages['5'] = fuzz.trimf(energy_beverages.universe, [240, 240, 270])
energy_beverages['6'] = fuzz.trimf(energy_beverages.universe, [270, 270, 300])
energy_beverages['7'] = fuzz.trimf(energy_beverages.universe, [300, 300, 330])
energy_beverages['8'] = fuzz.trimf(energy_beverages.universe, [330, 330, 360])
energy_beverages['9'] = fuzz.trimf(energy_beverages.universe, [360, 360, 390])
energy_beverages['10'] = fuzz.trimf(energy_beverages.universe, [390, 390, 391])

saturated_fats_beverages['0'] = fuzz.trimf(saturated_fats_beverages.universe, [0, 0, 1])
saturated_fats_beverages['1'] = fuzz.trimf(saturated_fats_beverages.universe, [1, 1, 2])
saturated_fats_beverages['2'] = fuzz.trimf(saturated_fats_beverages.universe, [2, 2, 3])
saturated_fats_beverages['3'] = fuzz.trimf(saturated_fats_beverages.universe, [3, 3, 4])
saturated_fats_beverages['4'] = fuzz.trimf(saturated_fats_beverages.universe, [4, 4, 5])
saturated_fats_beverages['5'] = fuzz.trimf(saturated_fats_beverages.universe, [5, 5, 6])
saturated_fats_beverages['6'] = fuzz.trimf(saturated_fats_beverages.universe, [6, 6, 7])
saturated_fats_beverages['7'] = fuzz.trimf(saturated_fats_beverages.universe, [7, 7, 8])
saturated_fats_beverages['8'] = fuzz.trimf(saturated_fats_beverages.universe, [8, 8, 9])
saturated_fats_beverages['9'] = fuzz.trimf(saturated_fats_beverages.universe, [9, 9, 10])
saturated_fats_beverages['10'] = fuzz.trimf(saturated_fats_beverages.universe, [10, 10, 11])

sugars_beverages['0'] = fuzz.trimf(sugars_beverages.universe, [0, 0, 0.5])
sugars_beverages['1'] = fuzz.trimf(sugars_beverages.universe, [0.5, 0.5, 2])
sugars_beverages['2'] = fuzz.trimf(sugars_beverages.universe, [2, 2, 3.5])
sugars_beverages['3'] = fuzz.trimf(sugars_beverages.universe, [3.5, 3.5, 5])
sugars_beverages['4'] = fuzz.trimf(sugars_beverages.universe, [5, 5, 6])
sugars_beverages['5'] = fuzz.trimf(sugars_beverages.universe, [6, 6, 7])
sugars_beverages['6'] = fuzz.trimf(sugars_beverages.universe, [7, 7, 8])
sugars_beverages['7'] = fuzz.trimf(sugars_beverages.universe, [8, 8, 9])
sugars_beverages['8'] = fuzz.trimf(sugars_beverages.universe, [9, 9, 10])
sugars_beverages['9'] = fuzz.trimf(sugars_beverages.universe, [10, 10, 11])
sugars_beverages['10'] = fuzz.trimf(sugars_beverages.universe, [11, 11, 20])

sodium_beverages['0'] = fuzz.trimf(sodium_beverages.universe, [0, 0, 0.2])
sodium_beverages['1'] = fuzz.trimf(sodium_beverages.universe, [0.2, 0.2, 0.4])
sodium_beverages['2'] = fuzz.trimf(sodium_beverages.universe, [0.4, 0.4, 0.6])
sodium_beverages['3'] = fuzz.trimf(sodium_beverages.universe, [0.6, 0.6, 0.8])
sodium_beverages['4'] = fuzz.trimf(sodium_beverages.universe, [0.8, 0.8, 1])
sodium_beverages['5'] = fuzz.trimf(sodium_beverages.universe, [1, 1, 1.2])
sodium_beverages['6'] = fuzz.trimf(sodium_beverages.universe, [1.2, 1.2, 1.4])
sodium_beverages['7'] = fuzz.trimf(sodium_beverages.universe, [1.4, 1.4, 1.6])
sodium_beverages['8'] = fuzz.trimf(sodium_beverages.universe, [1.6, 1.6, 1.8])
sodium_beverages['9'] = fuzz.trimf(sodium_beverages.universe, [1.8, 1.8, 2])
sodium_beverages['10'] = fuzz.trimf(sodium_beverages.universe, [2, 2, 2.2])
sodium_beverages['11'] = fuzz.trimf(sodium_beverages.universe, [2.2, 2.2, 2.4])
sodium_beverages['12'] = fuzz.trimf(sodium_beverages.universe, [2.4, 2.4, 2.6])
sodium_beverages['13'] = fuzz.trimf(sodium_beverages.universe, [2.6, 2.6, 2.8])
sodium_beverages['14'] = fuzz.trimf(sodium_beverages.universe, [2.8, 2.8, 3])
sodium_beverages['15'] = fuzz.trimf(sodium_beverages.universe, [3, 3, 3.2])
sodium_beverages['16'] = fuzz.trimf(sodium_beverages.universe, [3.2, 3.2, 3.4])
sodium_beverages['17'] = fuzz.trimf(sodium_beverages.universe, [3.4, 3.4, 3.6])
sodium_beverages['18'] = fuzz.trimf(sodium_beverages.universe, [3.6, 3.6, 3.8])
sodium_beverages['19'] = fuzz.trimf(sodium_beverages.universe, [3.8, 3.8, 4])
sodium_beverages['20'] = fuzz.trimf(sodium_beverages.universe, [4, 4, 5])

proteins_beverages['0'] = fuzz.trimf(proteins_beverages.universe, [0, 0, 1.2])
proteins_beverages['1'] = fuzz.trimf(proteins_beverages.universe, [1.2, 1.2, 1.5])
proteins_beverages['2'] = fuzz.trimf(proteins_beverages.universe, [1.5, 1.5, 1.8])
proteins_beverages['3'] = fuzz.trimf(proteins_beverages.universe, [1.8, 1.8, 2.1])
proteins_beverages['4'] = fuzz.trimf(proteins_beverages.universe, [2.1, 2.1, 2.4])
proteins_beverages['5'] = fuzz.trimf(proteins_beverages.universe, [2.4, 2.4, 2.7])
proteins_beverages['6'] = fuzz.trimf(proteins_beverages.universe, [2.7, 2.7, 3])
proteins_beverages['7'] = fuzz.trimf(proteins_beverages.universe, [3, 3, 4.5])

fibers_beverages['0'] = fuzz.trimf(fibers_beverages.universe, [0, 0, 3.0])
fibers_beverages['1'] = fuzz.trimf(fibers_beverages.universe, [3.0, 3.0, 4.1])
fibers_beverages['2'] = fuzz.trimf(fibers_beverages.universe, [4.1, 4.1, 5.2])
fibers_beverages['3'] = fuzz.trimf(fibers_beverages.universe, [5.2, 5.2, 6.3])
fibers_beverages['4'] = fuzz.trimf(fibers_beverages.universe, [6.3, 6.3, 7.4])
fibers_beverages['5'] = fuzz.trimf(fibers_beverages.universe, [7.4, 7.4, 8])
fibers_beverages['6'] = fuzz.trimf(fibers_beverages.universe, [8, 8, 9])  # Tambahan

fruits_vegetables_beverages['0'] = fuzz.trimf(fruits_vegetables_beverages.universe, [0, 0, 40])
fruits_vegetables_beverages['2'] = fuzz.trimf(fruits_vegetables_beverages.universe, [40, 40, 60])
fruits_vegetables_beverages['4'] = fuzz.trimf(fruits_vegetables_beverages.universe, [60, 60, 80])
fruits_vegetables_beverages['6'] = fuzz.trimf(fruits_vegetables_beverages.universe, [80, 80, 101])

# Define membership functions for Food
for i in range(11):
    n_points_food[str(i)] = fuzz.trimf(n_points_food.universe, [i*4, i*4, (i+1)*4])

for i in range(6):
    p_points_food[str(i)] = fuzz.trimf(p_points_food.universe, [i*3, i*3, (i+1)*3])

# Define membership functions for Food

for i in range(11):
    n_points_beverages[str(i)] = fuzz.trimf(n_points_beverages.universe, [i*4, i*4, (i+1)*4])

for i in range(7):
    p_points_beverages[str(i)] = fuzz.trimf(p_points_beverages.universe, [i*3, i*3, (i+1)*3])

# Define rules for N points for Food
n_rules_food = []
for i in range(11):
    n_rules_food.append(ctrl.Rule(energy[str(i)] | saturated_fats[str(i)] | sugars[str(i)] | sodium[str(i)], n_points_food[str(i)]))

# Define rules for P points for Food
p_rules_food = []
for i in range(6):
    p_rules_food.append(ctrl.Rule(proteins[str(i)] | fibers[str(i)], p_points_food[str(i)]))
    p_rules_food.append(ctrl.Rule(proteins['6'] | proteins['7'], p_points_food['5']))
    p_rules_food.append(ctrl.Rule(fruits_vegetables['0'], p_points_food['0']))
    p_rules_food.append(ctrl.Rule(fruits_vegetables['1'], p_points_food['1']))
    p_rules_food.append(ctrl.Rule(fruits_vegetables['2'], p_points_food['2']))
    p_rules_food.append(ctrl.Rule(fruits_vegetables['5'], p_points_food['5']))

# Define rules for N points for Beverages
n_rules_beverages = []
for i in range(11):
    n_rules_beverages.append(ctrl.Rule(energy_beverages[str(i)] | saturated_fats_beverages[str(i)] | sugars_beverages[str(i)] | sodium_beverages[str(i)], n_points_beverages[str(i)]))

# Define rules for P points
p_rules_beverages = []
for i in range(7):  # Sesuaikan dengan jumlah membership function baru
    p_rules_beverages.append(ctrl.Rule(proteins_beverages[str(i)] | fibers_beverages[str(i)], p_points_beverages[str(i)]))
p_rules_beverages.append(ctrl.Rule(proteins_beverages['6'] | proteins_beverages['7'], p_points_beverages['5']))
p_rules_beverages.append(ctrl.Rule(fruits_vegetables_beverages['0'], p_points_beverages['0']))
p_rules_beverages.append(ctrl.Rule(fruits_vegetables_beverages['2'], p_points_beverages['2']))
p_rules_beverages.append(ctrl.Rule(fruits_vegetables_beverages['4'], p_points_beverages['4']))
p_rules_beverages.append(ctrl.Rule(fruits_vegetables_beverages['6'], p_points_beverages['6']))

# Create and simulate the fuzzy control system
n_ctrl_food = ctrl.ControlSystem(n_rules_food)
p_ctrl_food = ctrl.ControlSystem(p_rules_food)
n_ctrl_beverages = ctrl.ControlSystem(n_rules_beverages)
p_ctrl_beverages = ctrl.ControlSystem(p_rules_beverages)

n_scoring_food = ctrl.ControlSystemSimulation(n_ctrl_food)
p_scoring_food = ctrl.ControlSystemSimulation(p_ctrl_food)
n_scoring_beverages = ctrl.ControlSystemSimulation(n_ctrl_beverages)
p_scoring_beverages = ctrl.ControlSystemSimulation(p_ctrl_beverages)

# Function to calculate Nutri-Score
def calculate_food_nutri_score(data):
    try:
        nutrition = data['nutritionFact']

        # Logic for food
        energy_val_kJ = nutrition['energy'] * 4.184  # Convert kCal to kJ
        sodium_val_mg = nutrition['sodium'] * 1000  # Convert g to mg

        n_scoring_food.input['energy'] = energy_val_kJ
        n_scoring_food.input['saturated_fats'] = nutrition['saturated_fat']
        n_scoring_food.input['sugars'] = nutrition['sugar']
        n_scoring_food.input['sodium'] = sodium_val_mg
        n_scoring_food.compute()

        p_scoring_food.input['proteins'] = nutrition['protein']
        p_scoring_food.input['fibers'] = nutrition['fiber']
        p_scoring_food.input['fruits_vegetables'] = nutrition['fruit_vegetable']
        p_scoring_food.compute()

        n_points_val = n_scoring_food.output['n_points_food']
        p_points_val = p_scoring_food.output['p_points_food']
        nutri_score = n_points_val - p_points_val

        # Determine Nutri-Score category
        if nutri_score <= 0:
            category_score = "1"
        elif nutri_score <= 2:
            category_score = "2"
        elif nutri_score <= 10:
            category_score = "3"
        elif nutri_score <= 18:
            category_score = "4"

        return {
            "nutri_score": nutri_score,
            "category": category_score
        }
    except Exception as e:
        return {"error": str(e)}
    
def calculate_beverages_nutri_score(data):
    try:
        nutrition = data['nutritionFact']    
            
        # Logic for beverages
        energy_val_kJ = nutrition['energy_beverages'] * 4.184  # Convert kCal to kJ
        sodium_val_mg = nutrition['sodium_beverages'] * 1000  # Convert g to mg

        n_scoring_beverages.input['energy_beverages'] = energy_val_kJ
        n_scoring_beverages.input['saturated_fats_beverages'] = nutrition['saturated_fat']
        n_scoring_beverages.input['sugars_beverages'] = nutrition['sugar']
        n_scoring_beverages.input['sodium_beverages'] = sodium_val_mg
        n_scoring_beverages.compute()

        p_scoring_beverages.input['proteins_beverages'] = nutrition['protein']
        p_scoring_beverages.input['fibers_beverages'] = nutrition['fiber']
        p_scoring_beverages.input['fruits_vegetables_beverages'] = nutrition['fruit_vegetable']
        p_scoring_beverages.compute()

        n_points_val = n_scoring_beverages.output['n_points_beverages']
        p_points_val = p_scoring_beverages.output['p_points_beverages']
        nutri_score = n_points_val - p_points_val

        # Determine Nutri-Score category
        if nutri_score <= 0.4:  # Special case for water
            category_score = "1"
        elif nutri_score <= 2:
            category_score = "2"
        elif nutri_score <= 6:
            category_score = "3"
        elif nutri_score <= 9:
            category_score = "4"
        else:
            category_score = "5"

        return {
            "nutri_score": nutri_score,
            "label_id": category_score
        }
    except Exception as e:
        return {"error": str(e)}

# Flask endpoint
# Endpoint 1: Calculate Nutri-Score for food
@app.route('/calculate-nutri-score/food', methods=['POST'])
def calculate_food():
    try:
        # Ambil data dari request body
        data_list = request.json  # Expecting an array of JSON objects

        if not isinstance(data_list, list):
            return jsonify({"error": "Input must be a list of food products"}), 400

        # Proses setiap produk
        results = []
        for data in data_list:
            result = calculate_food_nutri_score(data)
            results.append(result)

        return jsonify(results)  # Return all results as a list
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Endpoint 2: Calculate Nutri-Score for beverages
@app.route('/calculate-nutri-score/beverages', methods=['POST'])
def calculate_beverages():
    try:
        # Ambil data dari request body
        data_list = request.json  # Expecting an array of JSON objects

        if not isinstance(data_list, list):
            return jsonify({"error": "Input must be a list of beverages"}), 400

        # Proses setiap produk
        results = []
        for data in data_list:
            result = calculate_beverages_nutri_score(data)
            results.append(result)

        return jsonify(results)  # Return all results as a list
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run Flask app
if __name__ == "__main__":
    app.run(port=8080)