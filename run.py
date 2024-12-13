import streamlit as st
import random
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Genetic Algorithm - TV Show Ratings"
)

st.header("Genetic Algorithm - TV Show Ratings", divider="gray")

# Define constants and default values
POP_SIZE = 500
GENES = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
MUT_RATE = st.number_input("Enter your mutation rate", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
CO_RATE = st.number_input("Enter your crossover rate", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
TARGET = st.text_input("Enter your name", "Raja")

# Example ratings for demonstration purposes (use a dictionary or DataFrame to store actual ratings)
ratings_data = {
    'Program': ['TV9', 'Astro Arena Bola', 'Jurassic Park', 'Saw', 'Majalah 3'],
    '6 AM': [0.1, 0.0, 0.1, 0.2, 0.3],
    '7 AM': [0.1, 0.0, 0.1, 0.1, 0.4],
    '8 AM': [0.4, 0.0, 0.2, 0.1, 0.3],
    # Add more times as necessary
}

# Convert to DataFrame for easier handling
ratings_df = pd.DataFrame(ratings_data)

# Initialize population

def initialize_pop(TARGET):
    population = []
    tar_len = len(TARGET)
    for _ in range(POP_SIZE):
        temp = [random.choice(GENES) for _ in range(tar_len)]
        population.append(temp)
    return population

# Fitness calculation
def fitness_cal(TARGET, chromo_from_pop):
    difference = sum(1 for tar_char, chromo_char in zip(TARGET, chromo_from_pop) if tar_char != chromo_char)
    return [chromo_from_pop, difference]

# Selection
def selection(population):
    sorted_chromo_pop = sorted(population, key=lambda x: x[1])
    return sorted_chromo_pop[:POP_SIZE // 2]

# Crossover
def crossover(selected_chromo, CHROMO_LEN, population):
    offspring_cross = []
    for _ in range(POP_SIZE):
        parent1 = random.choice(selected_chromo)
        parent2 = random.choice(population[:POP_SIZE // 2])
        if random.random() < CO_RATE:
            crossover_point = random.randint(1, CHROMO_LEN - 1)
            child = parent1[0][:crossover_point] + parent2[0][crossover_point:]
        else:
            child = parent1[0][:]
        offspring_cross.append(child)
    return offspring_cross

# Mutation
def mutate(offspring, MUT_RATE):
    for arr in offspring:
        for i in range(len(arr)):
            if random.random() < MUT_RATE:
                arr[i] = random.choice(GENES)
    return offspring

# Replace function
def replace(new_gen, population):
    for i in range(len(population)):
        if population[i][1] > new_gen[i][1]:
            population[i] = new_gen[i]
    return population

# Main genetic algorithm function
def main(POP_SIZE, MUT_RATE, CO_RATE, TARGET, GENES):
    initial_population = initialize_pop(TARGET)
    found = False
    population = [fitness_cal(TARGET, chromo) for chromo in initial_population]
    generation = 1

    while not found:
        selected = selection(population)
        crossovered = crossover(selected, len(TARGET), population)
        mutated = mutate(crossovered, MUT_RATE)
        new_gen = [fitness_cal(TARGET, chromo) for chromo in mutated]

        population = replace(new_gen, population)
        population = sorted(population, key=lambda x: x[1])

        best_string = "".join(population[0][0])
        st.write(f"String: {population[0][0]} Generation: {generation} Fitness: {population[0][1]}")

        if population[0][1] == 0:
            st.write("Target found")
            break

        generation += 1

    # Display highest rating program per time slot
    st.subheader("Highest Rating Program per Time Slot")
    max_ratings = ratings_df.set_index('Program').idxmax()
    max_values = ratings_df.set_index('Program').max()
    summary_df = pd.DataFrame({'Time Slot': max_ratings.index, 'Program': max_ratings.values, 'Rating': max_values.values})
    st.write(summary_df)

# Run calculation if button is pressed
if st.button("Calculate"):
    main(POP_SIZE, MUT_RATE, CO_RATE, TARGET, GENES)
