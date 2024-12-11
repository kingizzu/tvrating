import streamlit as st
import csv
import random
import pandas as pd

st.set_page_config(
    page_title="TV Ratings by Raja Izzudin"
)

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

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)

##################################### DEFINING PARAMETERS AND DATASET ################################################################
# Sample rating programs dataset for each time slot.
ratings = program_ratings_dict

GEN = 100
POP = 50
EL_S = 2  # elitism size

all_programs = list(ratings.keys())  # all programs
all_time_slots = list(range(6, 24))  # time slots

######################################### DEFINING FUNCTIONS ########################################################################
# defining fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

# initializing the population
def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]

    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)

    return all_schedules

# selection
def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0

    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule

    return best_schedule

############################################# GENETIC ALGORITHM #############################################################################

# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# mutating
def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

# Genetic algorithms with parameters
def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=0.8, mutation_rate=0.2, elitism_size=EL_S):
    population = [initial_schedule]

    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []

        # Elitism
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    return population[0]

##################################################### MAIN LOGIC ###################################################################################

# User inputs for crossover and mutation rates
CO_R = st.number_input("Crossover Rate", min_value=0.0, max_value=0.95, value=0.8, step=0.01)  # Allows numeric input
MUT_R = st.number_input("Mutation Rate", min_value=0.01, max_value=0.05, value=0.02, step=0.01)  # Allows numeric input

# Execute button
if st.button("Run"):
    # brute force
    all_possible_schedules = initialize_pop(all_programs, all_time_slots)
    initial_best_schedule = finding_best_schedule(all_possible_schedules)

    rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
    genetic_schedule = genetic_algorithm(
        initial_best_schedule,
        generations=GEN,
        population_size=POP,
        crossover_rate=CO_R,
        mutation_rate=MUT_R,
        elitism_size=EL_S
    )

    final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]

    # Prepare data for the table
    schedule_data = {
        "Time Slot": [f"{time_slot:02d}:00" for time_slot in all_time_slots],
        "Program": [""] * len(all_time_slots),
        "Rating": [""] * len(all_time_slots),
    }

    # Fill the schedule data
    for time_slot, program in zip(all_time_slots, final_schedule):
        schedule_data["Program"][time_slot - 6] = program  # Adjust index by the starting time (6:00)
        schedule_data["Rating"][time_slot - 6] = ratings[program][time_slot - 6]  # Get rating for the time slot

    # Convert to DataFrame
    schedule_df = pd.DataFrame(schedule_data)

    # Display the schedule as a table in Streamlit
    st.subheader("TV Ratings based on Mutation and Crossover")
    st.write(schedule_df)

    # Display total ratings for the final schedule
    total_ratings = fitness_function(final_schedule)
    st.write("Total Ratings for the Final Schedule:", total_ratings)
