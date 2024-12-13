import streamlit as st
import csv
import random
import pandas as pd

st.set_page_config(page_title="TV Ratings by Raja Izzudin")
st.header("TV Ratings by Raja Izzudin", divider="gray")

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header
        header = next(reader)
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
            program_ratings[program] = ratings
    return program_ratings

# Path to the CSV file
file_path = 'program_ratings.csv'
program_ratings_dict = read_csv_to_dict(file_path)

# Setup parameters
GEN = 50  # Reduced generations for faster execution
POP = 20  # Reduced population size for efficiency
EL_S = 2  # Elitism size

all_programs = list(program_ratings_dict.keys())  # All programs
all_time_slots = list(range(6, 24))  # Time slots

# Fitness function to evaluate the total ratings of a schedule
def fitness_function(schedule):
    return sum(program_ratings_dict[program][time_slot] for time_slot, program in enumerate(schedule))

# Initialize population with random unique programs
def initialize_pop(programs, time_slots, population_size):
    population = []
    for _ in range(population_size):
        schedule = random.sample(programs, time_slots)
        population.append(schedule)
    return population

# Crossover function to combine two parent schedules
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 1)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutation function that changes one program in the schedule
def mutate(schedule, programs):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(programs)
    
    # Ensure the program being mutated is not already in the schedule
    while new_program in schedule:
        new_program = random.choice(programs)

    schedule[mutation_point] = new_program
    return schedule

# Genetic algorithm implementation
def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=0.8, mutation_rate=0.2, elitism_size=EL_S):
    population = initialize_pop(all_programs, len(initial_schedule), population_size)
    population.append(initial_schedule)  # Ensure the initial schedule is always included

    for generation in range(generations):
        # Evaluate fitness of each individual
        population.sort(key=fitness_function, reverse=True)
        
        new_population = population[:elitism_size]  # Elitism

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]  # Clone parents if no crossover
            
            if random.random() < mutation_rate:
                child1 = mutate(child1, all_programs)
            if random.random() < mutation_rate:
                child2 = mutate(child2, all_programs)

            new_population.extend([child1, child2])

        population = new_population[:population_size]  # Trim to population size

    return max(population, key=fitness_function)  # Return the best schedule found

# User inputs for crossover and mutation rates
CO_R = st.number_input("Crossover Rate", min_value=0.0, max_value=0.95, value=0.8, step=0.01)
MUT_R = st.number_input("Mutation Rate", min_value=0.01, max_value=0.5, value=0.2, step=0.01)

# Execute button
if st.button("Run"):
    # Use a simple initial schedule
    initial_best_schedule = random.sample(all_programs, len(all_time_slots))
    
    final_schedule = genetic_algorithm(
        initial_best_schedule,
        generations=GEN,
        population_size=POP,
        crossover_rate=CO_R,
        mutation_rate=MUT_R,
        elitism_size=EL_S
    )

    # Prepare data for the table
    schedule_data = {
        "Time Slot": [f"{time_slot:02d}:00" for time_slot in all_time_slots],
        "Program": [""] * len(all_time_slots),
        "Rating": [""] * len(all_time_slots),
    }

    for time_slot, program in zip(all_time_slots, final_schedule):
        schedule_data["Program"][time_slot - 6] = program  # Adjust index by the starting time (6:00)
        schedule_data["Rating"][time_slot - 6] = program_ratings_dict[program][time_slot - 6]  # Get rating for the slot

    # Convert to DataFrame
    schedule_df = pd.DataFrame(schedule_data)

    # Display the schedule as a table in Streamlit
    st.subheader("TV Ratings based on Mutation and Crossover")
    st.write(schedule_df)

    # Display total ratings for the final schedule
    total_ratings = fitness_function(final_schedule)
    st.write("Total Ratings for the Final Schedule:", total_ratings)
